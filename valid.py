
import torch
from nltk.translate.bleu_score import sentence_bleu

from datareading import output_lang, EOS_token, SOS_token, device, input_lang, MAX_LENGTH, test_pairs, val_pairs, train_pairs
from net import AttnDecoderRNN, tensorFromSentence


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        # 将输入句子转换为Tensor对象
        input_tensor = tensorFromSentence(input_lang, [sentence])
        input_length = input_tensor.size()[0]
        # 将输入句子输入编码器，并获取编码器的输出和隐藏状态
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
            # 初始化解码器的输入和隐藏状态
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        # 存储解码器的输出和注意力权重
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        # 解码器每次解码一个时间步
        for di in range(max_length):
            if isinstance(decoder, AttnDecoderRNN):  # 检查 decoder 是否为 AttnDecoderRNN 的实例
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                decoder_attentions[di] = torch.zeros(max_length)  # 如果没有attention,attention值置为0
            # 选择概率最大的词作为解码器的输出
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            # 将解码器的输出作为下一个时间步的输入
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pair, n=5, prints = False):
    total_bleu_score = 0
    all_bleu_scores = []
    num_sentences = len(pair)
    for i in range(num_sentences):
        # 获取句子对
        input_sentence = pair[i][0]
        target_sentence = pair[i][1]
        # 进行翻译
        output_words, attentions = evaluate(encoder, decoder, input_sentence[0])
        # 如果是选中的句子则打印
        if i < n and prints:
            print('中文原句', input_sentence)
            print('英文原句', target_sentence)
            output_sentence = ' '.join(output_words)
            print('译句', output_sentence)
            print('')

        # 计算BLEU得分
        reference = [[str(word) for word in sentence] for sentence in target_sentence]
        candidate = output_words[:-1]
        bleu_score = sentence_bleu(reference, candidate)
        total_bleu_score += bleu_score
        all_bleu_scores.append(bleu_score)
    # 计算平均BLEU得分
    average_bleu_score = total_bleu_score / num_sentences
    print('Average Test BLEU Score:', average_bleu_score)
    return (total_bleu_score, average_bleu_score)




