import os
import random
import torch
import time
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
from torch import optim, nn

from datareading import EOS_token, device,  SOS_token, MAX_LENGTH, train_pairs, val_pairs, \
    test_pairs
from net import AttnDecoderRNN, tensorsFromPair

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: 将目标作为下一个输入
        for di in range(target_length):
            if isinstance(decoder, AttnDecoderRNN):  # 检查 decoder 是否为 AttnDecoderRNN 的实例
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # 不适用 teacher forcing: 使用自己的预测作为下一个输入
        for di in range(target_length):
            if isinstance(decoder, AttnDecoderRNN):  # 检查 decoder 是否为 AttnDecoderRNN 的实例
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, checkpoint_path, print_every=1000, plot_every=100, learning_rate=0.01):
    start_iteration = load_checkpoint(encoder, decoder, checkpoint_path)
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    train_input_pairs = [tensorsFromPair(pair) for pair in train_pairs]
    val_input_pairs = [tensorsFromPair(pair) for pair in val_pairs]
    test_input_pairs = [tensorsFromPair(pair) for pair in test_pairs]
    total_input_pairs = train_input_pairs + val_input_pairs + test_input_pairs
    n_iters = len(total_input_pairs)
    criterion = nn.NLLLoss()
    for iter in range(start_iteration + 1, n_iters + 1):
        if iter <= len(train_input_pairs):
            training_pair = train_input_pairs[iter - 1]
        elif iter <= len(train_input_pairs) + len(val_input_pairs):
            training_pair = val_input_pairs[iter - len(train_input_pairs) - 1]
        else:
            training_pair = test_input_pairs[iter - len(train_input_pairs) - len(val_input_pairs) - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            train_loss = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d  %d%%) Loss:%.4f PPL:%.4f' % (timeSince(start, iter / n_iters),
                                          iter, iter / n_iters * 100, train_loss, math.exp(train_loss)))
            # 保存检查点
            save_checkpoint(encoder, decoder, iter, checkpoint_path)
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_losses.append(math.exp(plot_loss_avg))
            plot_loss_total = 0
    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 该定时器用于定时记录时间
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def load_checkpoint(encoder, decoder, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        return checkpoint['iteration']
    else:
        return 0

def save_checkpoint(encoder, decoder, iteration, checkpoint_path):
    torch.save({
        'iteration': iteration,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, checkpoint_path)