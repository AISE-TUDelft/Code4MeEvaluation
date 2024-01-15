from typing import List, Callable
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
import torch


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_().to(device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0).to(device)]
        self.nextYs[0][:] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] in self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode='trunc')
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] in self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] in self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] not in self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                tokens.append(tok)
                if tok in self._eos:
                    break
            sentence.append(tokens)
        return sentence


device = torch.device("cuda")
m = torch.nn.LogSoftmax(dim=-1).to(device)
zero = torch.cuda.LongTensor(1).fill_(0).to(device)


def create_predict_fn(checkpoint_path_or_url: str, is_java: bool = False) -> Callable[[str, str], str]:
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path_or_url, do_lower_case=False, sep_token='<EOL>',
                                              bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                              unk_token='<|UNKNOWN|>')
    if is_java:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{'), tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.sep_token_id]
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path_or_url)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id]
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    def codegpt_predict(left_context: str) -> str:
        if not is_java:
            # java version did not train on data with <EOL>
            left_context = left_context.replace("\n", "<EOL>")
        input_size = 896
        predict_size = 128
        block_size = input_size + predict_size

        input = left_context

        tokens = tokenizer.encode(input)[-(input_size - 1):]
        # print tokens as strings
        # prepend with <s>
        tokens = [tokenizer.bos_token_id] + tokens
        inputs = torch.tensor(tokens, device=device).unsqueeze(0)
        with torch.no_grad():
            beam_size = 1
            outputs = model(inputs[:, :-1])[1]
            p = []
            for i in range(inputs.shape[0]):
                past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                        outputs]
                past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, inputs[i][-1].data, break_ids)
                input_ids = None
                for _ in range(predict_size):
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    outputs = model(input_ids, past_key_values=past_hidden)
                    out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                            outputs[1]]
                    past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                for break_id in break_ids:
                    if break_id in t:
                        t = t[:t.index(break_id)]
                if is_java:
                    text = DecodeIds(t).strip("{").strip()
                else:
                    text = DecodeIds(t).strip("<EOL>").strip()
                return text
        return ""

    return codegpt_predict
