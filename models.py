import os
import torch
from typing import Tuple, List
import torch.nn as nn
from transformers import AutoModel


class BertCrf(nn.Module):
    def __init__(self, num_labels: int, bert_name: str, dropout: float = 0.1, use_crf: bool = True):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(self.bert.config.hidden_size, num_labels)
        if self.use_crf:
            self.start_transitions = nn.Parameter(torch.empty(num_labels))
            self.end_transitions = nn.Parameter(torch.empty(num_labels))
            self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                confidence: torch.Tensor = None) -> torch.Tensor:
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()
        if self.use_crf:
            features = torch.swapaxes(features, 0, 1)
            attention_mask = torch.swapaxes(attention_mask, 0, 1)
            labels = torch.swapaxes(labels, 0, 1)
            labels = torch.where(labels == -100, torch.tensor(0).to(labels.device), labels)
            log_numerator = self._compute_log_numerator(features=features, labels=labels, mask=attention_mask)
            log_denominator = self._compute_log_denominator(features=features, mask=attention_mask)
            loss = log_denominator - log_numerator
            if confidence is not None:
                loss = loss * confidence
            return torch.mean(loss)
        else:
            loss = self.cross_entropy(
                features.view(-1, self.num_labels),
                labels.view(-1),
            )
            if confidence is not None:
                loss = loss * confidence.view(-1)
            loss = torch.masked_select(loss, labels.view(-1) != -100)
            return torch.mean(loss)

    def _compute_log_denominator(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[0]
        log_score_over_all_seq = self.start_transitions + features[0]
        for i in range(1, seq_len):
            next_log_score_over_all_seq = torch.logsumexp(
                log_score_over_all_seq.unsqueeze(2) + self.transitions + features[i].unsqueeze(1),
                dim=1,
            )
            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
        log_score_over_all_seq += self.end_transitions
        return torch.logsumexp(log_score_over_all_seq, dim=1)

    def _compute_log_numerator(self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, bs, _ = features.shape
        score_over_seq = self.start_transitions[labels[0]] + features[0, torch.arange(bs), labels[0]]
        for i in range(1, seq_len):
            score_over_seq += (self.transitions[labels[i - 1], labels[i]] + features[i, torch.arange(bs), labels[i]]) * mask[i]
        seq_lens = mask.sum(dim=0) - 1
        last_tags = labels[seq_lens.long(), torch.arange(bs)]
        score_over_seq += self.end_transitions[last_tags]
        return score_over_seq

    def get_bert_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        hidden = self.dropout(hidden)
        return self.hidden2label(hidden), hidden

    def _viterbi_decode(self, features: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        seq_len, bs, _ = features.shape
        log_score_over_all_seq = self.start_transitions + features[0]
        backpointers = torch.empty_like(features)
        for i in range(1, seq_len):
            next_log_score_over_all_seq = (
                    log_score_over_all_seq.unsqueeze(2) + self.transitions + features[i].unsqueeze(1)
            )
            next_log_score_over_all_seq, indices = next_log_score_over_all_seq.max(dim=1)
            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
            backpointers[i] = indices
        backpointers = backpointers[1:].int()
        log_score_over_all_seq += self.end_transitions
        seq_lens = mask.sum(dim=0) - 1
        best_paths = []
        for seq_ind in range(bs):
            best_label_id = torch.argmax(log_score_over_all_seq[seq_ind]).item()
            best_path = [best_label_id]
            for backpointer in reversed(backpointers[: seq_lens[seq_ind]]):
                best_path.append(backpointer[seq_ind][best_path[-1]].item())
            best_path.reverse()
            best_paths.append(best_path)
        return best_paths

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[int]]:
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()
        if self.use_crf:
            features = torch.swapaxes(features, 0, 1)
            mask = torch.swapaxes(attention_mask, 0, 1)
            return self._viterbi_decode(features=features, mask=mask)
        else:
            labels = torch.argmax(features, dim=2)
            predictions = []
            for i in range(len(labels)):
                predictions.append(labels[i][attention_mask[i]].tolist())
            return predictions

    def decode_with_confidence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[
        List[List[int]], List[List[float]]]:
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()
        if self.use_crf:
            features = torch.swapaxes(features, 0, 1)
            mask = torch.swapaxes(attention_mask, 0, 1)
            predictions = self._viterbi_decode(features=features, mask=mask)
            confidences = []
            for seq_idx, path in enumerate(predictions):
                seq_confidences = []
                for i, label in enumerate(path):
                    logits = features[i, seq_idx, :]
                    probabilities = torch.softmax(logits, dim=-1)
                    seq_confidences.append(probabilities[label].item())
                confidences.append(seq_confidences)
            return predictions, confidences
        else:
            logits = torch.softmax(features, dim=2)
            labels = torch.argmax(features, dim=2)
            predictions, confidences = [], []
            for i in range(len(labels)):
                seq_predictions = labels[i][attention_mask[i]].tolist()
                seq_confidences = logits[i][attention_mask[i], seq_predictions].tolist()
                predictions.append(seq_predictions)
                confidences.append(seq_confidences)
            return predictions, confidences

    def save_to(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_from(self, path: str):
        self.load_state_dict(torch.load(path))
