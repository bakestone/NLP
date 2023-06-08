from matplotlib import pyplot as plt, ticker
import locale
locale.setlocale(locale.LC_ALL, 'chinese')
from datareading import input_lang, device, output_lang, MAX_LENGTH, test_pairs
from net import EncoderRNN, AttnDecoderRNN, DecoderRNN
from train import trainIters
from valid import evaluateRandomly, evaluate
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"   # 根据实际编码设置

hidden_size = 256
#iters = 75000
#prints = 5000
iters = 50000
print_every = iters/10


encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, iters, checkpoint_path = 'savepoint/sample = 1w (reduce/r = 1 (iters = 1w)/checkpoint_rnn.pth', print_every = print_every)
trainIters(encoder1, attn_decoder1, iters, checkpoint_path = 'savepoint/sample = 1w (reduce/r = 1 (iters = 1w)/checkpoint_attn.pth', print_every = print_every)


evaluateRandomly(encoder1, decoder1, pair = test_pairs , prints = True)
print('-' * 100)
evaluateRandomly(encoder1, attn_decoder1, pair = test_pairs, prints = True)

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar

    attentions = np.array(attentions[:,:10])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='cool')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))

    showAttention(input_sentence, output_words, attentions)

evaluateAndShowAttention("他们 蔑视 当权 人士 ")
# evaluateAndShowAttention("她 试图 成为 法兰西共和国 历史 上 首位 女性 总统 ")
# evaluateAndShowAttention("它们 日益 坚信 在 任何 不 对称 的 国家 间 冲突 中 不论 武装 叛乱者 有 多 强大 其 宗教 驱动力 有 多 深厚 胜利 都 是 不可捉摸 的 ")
# evaluateAndShowAttention("我们 正 见证 着 各种 同时 进行 中 的 全球 收敛 现象 各国 之间 收入 财富 和 知识 差距 的 不断 收窄 以及 本地 分化 现象 各国 内部 收入 财富 和 知识 差距 的 不断 加大 ")
