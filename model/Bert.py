from transformers import BertModel, BertConfig,logging
logging.set_verbosity_error()
import torch.nn as nn

# ============================================================
# Text Backbone: BERT Encoder
# 文本主干网络 —— 对应论文 Section 3.1
# ============================================================
class Bert_Model(nn.Module):
    """
    BERT encoder for textual feature extraction.

    Paper description:
        - We adopt BERT as the text encoder.
        - We extract representations from the last four layers.
        - CLS tokens are used for hierarchical attention.
        - Token features are used for weighted aggregation.

    Outputs:
        CLS embedding
        Last 4 layer CLS representations
        Last 4 layer token representations
    """
    def __init__(self,config):
        super(Bert_Model,self).__init__()

        # Load BERT configuration
        self.bert_config = BertConfig.from_pretrained(config.Bert_path)

        # Enable hidden states output
        self.bert_config.output_attentions=False
        self.bert_config.output_hidden_states=True

        # Load pretrained BERT weights
        self.bert = BertModel.from_pretrained(
            config.Bert_path,
            config=self.bert_config)


    def forward(self, input_id, token_type_ids, mask):
        """
        Forward pass of BERT encoder.

        Inputs:
            input_id       : token ids
            token_type_ids : segment ids
            mask           : attention mask

        Returns:
            pooled_output  : CLS embedding from last layer
            layer3         : CLS from last 4 layers
            hidden3        : token features from last 4 layers
        """

        # Standard BERT forward
        output = self.bert(input_ids=input_id,
                           token_type_ids=token_type_ids,
                           attention_mask=mask,
                           return_dict=False)

        # hidden_states: tuple of all layer outputs
        hidden_states = output[2]

        # ====================================================
        # Extract last 4 layers (Layer 9~12)
        # 提取最后四层特征 —— 用于Hierarchical Attention
        # ====================================================

        # CLS representations (for generating attention weights)
        layer3 = (hidden_states[9][:, 0, :].unsqueeze(1),
                  hidden_states[10][:, 0, :].unsqueeze(1),
                  hidden_states[11][:, 0, :].unsqueeze(1),
                  hidden_states[12][:, 0, :].unsqueeze(1))

        # Token representations (exclude CLS and SEP)
        hidden3 = (hidden_states[9][:,1:-1,:].unsqueeze(1),
                   hidden_states[10][:,1:-1,:].unsqueeze(1),
                  hidden_states[11][:,1:-1,:].unsqueeze(1),
                   hidden_states[12][:,1:-1,:].unsqueeze(1))

        # output[1] is pooled CLS embedding
        return output[1],layer3,hidden3