import configargparse
import torch

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from utils import add_sos_eos, LabelSmoothingLoss


class ASRModel(torch.nn.Module):
    def __init__(self, params: configargparse.Namespace):
        """E2E ASR model implementation.

        Args:
            params: The training options
        """
        super().__init__()

        self.ignore_id = params.text_pad
        self.sos = params.odim - 1
        self.eos = self.sos

        self.encoder = TransformerEncoder(
            input_size=params.idim,
            output_size=params.hidden_dim,
            attention_heads=params.attention_heads,
            linear_units=params.linear_units,
            num_blocks=params.eblocks,
            dropout_rate=params.edropout,
            positional_dropout_rate=params.edropout,
            attention_dropout_rate=params.edropout,
        )
        self.decoder = TransformerDecoder(
            vocab_size=params.odim,
            encoder_output_size=params.hidden_dim,
            attention_heads=params.attention_heads,
            linear_units=params.linear_units,
            num_blocks=params.dblocks,
            dropout_rate=params.ddropout,
            positional_dropout_rate=params.ddropout,
            self_attention_dropout_rate=params.ddropout,
            src_attention_dropout_rate=params.ddropout,
        )
        self.criterion_att = LabelSmoothingLoss(
            size=params.odim,
            padding_idx=self.ignore_id,
            smoothing=params.label_smoothing,
            normalize_length=True,
        )

    def forward(
        self,
        xs,
        xlens,
        ys,
        ylens,
    ):
        """Forward propogation for ASRModel

        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences
        :params torch.LongTensor ys_ref- Padded Text Tokens
        :params list ylen- Lengths of unpadded text sequences
        """
        xlens = torch.tensor(xlens, dtype=torch.long, device=xs.device)
        ylens = torch.tensor(ylens, dtype=torch.long, device=xs.device)

        # TODO: implement forward of the ASR model

        # 1. Encoder forward (CNN + Transformer)
        encoder_out, encoder_out_lens = self.encoder(xs, xlens)
        
        # 2. Compute Loss by calling self.calculate_loss()
        loss = self.calculate_loss(encoder_out, encoder_out_lens, ys, ylens)
        
        return loss

    def calculate_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        
        # 1. Forward decoder
        decoded_token_score = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)
        
        # 2. Compute attention loss using self.criterion_att()
        loss_att = self.criterion_att(decoded_token_score, ys_out_pad)

        return loss_att

    def decode_greedy(self, xs, xlens):
        """Perform Greedy Decoding using trained ASRModel

        :params torch.Tensor xs- Speech feature input, (batch, time, dim)
        :params list xlens- Lengths of unpadded feature sequences, (batch,)
        """

        xlens = torch.tensor(xlens, dtype=torch.long, device=xs.device)

        # Encoder forward (CNN + Transformer)
        h, h_lens = self.encoder(xs, xlens)
        
        # TODO: implement greedy decoding
        # Hints:
        # - Start from <sos> and predict new tokens step-by-step until <eos>. You need a loop.
        # - You may need to set a maximum decoding length.
        # - You can use self.decoder.forward_one_step() for each step which has caches

        # predictions: list of lists of token_ints, e.g.:
        # [[23, 33, 111], [24, 67, 99, 199], ...]

        '''
        # maximum decoding length
        max_decoding_length = 500
        
        
        # token sequence, initialize with <sos> (size: batch x max_decoding)
        token_sequence = torch.zeros(len(h_lens), max_decoding_length, dtype=torch.long, device='cuda')
        print("ys_in_pad:", token_sequence.size())
        token_sequence[:, 0] = self.sos
        
        # sequence length array (batch)
        token_sequence_lens = torch.ones(len(h_lens), dtype=torch.long)
        print("ys_in_len:", token_sequence_lens.size())
        # keep a list of predictions
        predictions = [[[self.sos]] * h.size()[0]]
        
        for batch in range(h.size()[0]):
            cache = None
            for i in range(max_decoding_length):
                # compute probabilities for possible next tokens
                h_batch = h[batch, :, :].unsqueeze(0)
                token_batch = token_sequence[batch, :i+1].unsqueeze(0)
                print("ys_in_pad batch:", token_batch.size())
                y_hat, cache = self.decoder.forward_one_step(h_batch, h_lens, token_batch, token_sequence_lens, cache)

                # predict the next token
                predicted_token = torch.argmax(y_hat, axis=1)

                # add predicted token to list of predictions
                predictions[batch].append(predicted_token.item())
                
                token_sequence[batch, i] = predicted_token.item()

                token_sequence_lens[batch] += 1
                
                print("predictions list:", predictions)
                # Break the loop if <eos> is predicted
                if predicted_token.item() == self.eos:
                    break
        return predictions
        '''
        # maximum decoding length
        max_decoding_length = 150
        num_batches = xs.size(0)
        
        # token sequence, initialize with <sos> 
        token_sequence = torch.full((num_batches, 1), fill_value=self.sos, dtype=torch.long, device=xs.device)
        
        # sequence length array (batch)
        token_sequence_lens = torch.ones(num_batches, dtype=torch.long, device=xs.device)
        
        predictions = [[[]] * num_batches][0]
        
        cache = None
        
        for i in range(max_decoding_length):
            y_hat, cache = self.decoder.forward_one_step(h, h_lens, token_sequence, token_sequence_lens, cache)
            
            # predict the next token
            predicted_token = torch.argmax(y_hat, dim=-1)
           
            for batch in range(num_batches):
                predictions[batch].append(predicted_token[batch].item())
                token_sequence_lens[batch] = token_sequence_lens[batch] + 1
                    
            # add predicted tokens to token sequences of all batches
            token_sequence = torch.cat([token_sequence, predicted_token.unsqueeze(1)], dim=1)
            
        # clean predictions, everything after eos is deleted
        cleaned_predictions = []
        for i in range(len(predictions)):
            processed_pred = []
            for token in predictions[i]:
                if token != self.eos:
                    processed_pred.append(token)
                else:
                    print("reached eos")
                    break
            cleaned_predictions.append(processed_pred)
        print(cleaned_predictions)
        return cleaned_predictions