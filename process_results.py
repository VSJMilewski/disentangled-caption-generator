import numpy as np
import pickle
import matplotlib.pyplot as plt

# set colors globally
color1 = '#396ab1'
color2 = '#da7c30'
color3 = '#922428'


def lossdict_to_lists(losses):
    k = sorted([k for k in losses.keys()])
    v = [losses[i] for i in k]
    return k, v


def my_ceil(x, base=5):
    return int(base * np.ceil(float(x)/base))


def create_losses_figure(train, avg, val, min_, max_, filename):
    avg_k, avg_v = lossdict_to_lists(avg)
    val_k, val_v = lossdict_to_lists(val)
    plt.figure()
    plt.grid(True)
    plt.plot(range(1, len(train) + 1), train, color=color1, linewidth=0.5, label='Batch Train Loss')
    plt.plot(avg_k, avg_v, color=color2, linewidth=2, label='Epoch Train Loss')
    plt.plot(val_k, val_v, color=color3, linewidth=2, label='Epoch Validation Loss')
    plt.ylim(min_, max_)
    plt.xlim(0, max(avg_k) + 1)
    plt.xticks(([0] + avg_k)[0::5], range(0, my_ceil(len(avg_k)))[0::5])
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')


adam_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad5.0/losses_train.pkl', 'rb'))
adagrad_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdagrad_grad5.0/losses_train.pkl', 'rb'))
rmsprop_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optRMSProp_grad5.0/losses_train.pkl', 'rb'))
clip2_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad2.5/losses_train.pkl', 'rb'))
clip7_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad7.5/losses_train.pkl', 'rb'))
clip10_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad10.0/losses_train.pkl', 'rb'))
drop6_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.6_optAdam_grad5.0/losses_train.pkl', 'rb'))
drop7_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.7_optAdam_grad5.0/losses_train.pkl', 'rb'))
drop8_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.8_optAdam_grad5.0/losses_train.pkl', 'rb'))
drop9_trainloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.9_optAdam_grad5.0/losses_train.pkl', 'rb'))

adam_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad5.0/losses_train_avg.pkl', 'rb'))
adagrad_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdagrad_grad5.0/losses_train_avg.pkl', 'rb'))
rmsprop_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optRMSProp_grad5.0/losses_train_avg.pkl', 'rb'))
clip2_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad2.5/losses_train_avg.pkl', 'rb'))
clip7_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad7.5/losses_train_avg.pkl', 'rb'))
clip10_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad10.0/losses_train_avg.pkl', 'rb'))
drop6_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.6_optAdam_grad5.0/losses_train_avg.pkl', 'rb'))
drop7_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.7_optAdam_grad5.0/losses_train_avg.pkl', 'rb'))
drop8_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.8_optAdam_grad5.0/losses_train_avg.pkl', 'rb'))
drop9_avgloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.9_optAdam_grad5.0/losses_train_avg.pkl', 'rb'))

adam_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad5.0/losses_eval.pkl', 'rb'))
adagrad_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdagrad_grad5.0/losses_eval.pkl', 'rb'))
rmsprop_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optRMSProp_grad5.0/losses_eval.pkl', 'rb'))
clip2_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad2.5/losses_eval.pkl', 'rb'))
clip7_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad7.5/losses_eval.pkl', 'rb'))
clip10_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.5_optAdam_grad10.0/losses_eval.pkl', 'rb'))
drop6_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.6_optAdam_grad5.0/losses_eval.pkl', 'rb'))
drop7_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.7_optAdam_grad5.0/losses_eval.pkl', 'rb'))
drop8_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.8_optAdam_grad5.0/losses_eval.pkl', 'rb'))
drop9_valloss = pickle.load(open('output/BASELINE_flickr8k_beam1_lstm2_pat10_emb128_hidden512_p0.9_optAdam_grad5.0/losses_eval.pkl', 'rb'))


######################################
# process the optimizer tuning results
######################################
min_loss = np.floor((np.min([np.min(adam_trainloss), np.min(adagrad_trainloss), np.min(rmsprop_trainloss)])))
max_loss = np.ceil((np.max([np.max(adam_trainloss), np.max(adagrad_trainloss), np.max(rmsprop_trainloss)])))

# create the default reference
create_losses_figure(adam_trainloss, adam_avgloss, adam_valloss, min_loss, max_loss, 'plots/tune_default.png')
create_losses_figure(adagrad_trainloss, adagrad_avgloss, adagrad_valloss, min_loss, max_loss, 'plots/tune_adagrad.png')
create_losses_figure(rmsprop_trainloss, rmsprop_avgloss, rmsprop_valloss, min_loss, max_loss, 'plots/tune_rmsprop.png')

######################################
# process the grad clipping tuning
######################################
min_loss = np.floor((np.min([np.min(adam_trainloss), np.min(clip2_trainloss),
                             np.min(clip7_trainloss), np.min(clip10_trainloss)])))
max_loss = np.floor((np.max([np.max(adam_trainloss), np.max(clip2_trainloss),
                             np.max(clip7_trainloss), np.max(clip10_trainloss)])))
create_losses_figure(clip2_trainloss, clip2_avgloss, clip2_valloss, min_loss, max_loss, 'plots/tune_clip2.png')
create_losses_figure(clip7_trainloss, clip7_avgloss, clip7_valloss, min_loss, max_loss, 'plots/tune_clip7.png')
create_losses_figure(clip10_trainloss, clip10_avgloss, clip10_valloss, min_loss, max_loss, 'plots/tune_clip10.png')

######################################
# process the dropout tuning
######################################
min_loss = np.floor((np.min([np.min(adam_trainloss), np.min(drop6_trainloss), np.min(drop7_trainloss),
                             np.min(drop8_trainloss), np.min(drop9_trainloss)])))
max_loss = np.floor((np.max([np.max(adam_trainloss), np.max(drop6_trainloss), np.max(drop7_trainloss),
                             np.max(drop8_trainloss), np.max(drop9_trainloss)])))
create_losses_figure(drop6_trainloss, drop6_avgloss, drop6_valloss, min_loss, max_loss, 'plots/tune_drop6.png')
create_losses_figure(drop7_trainloss, drop7_avgloss, drop7_valloss, min_loss, max_loss, 'plots/tune_drop7.png')
create_losses_figure(drop8_trainloss, drop8_avgloss, drop8_valloss, min_loss, max_loss, 'plots/tune_drop8.png')
create_losses_figure(drop9_trainloss, drop9_avgloss, drop9_valloss, min_loss, max_loss, 'plots/tune_drop9.png')
