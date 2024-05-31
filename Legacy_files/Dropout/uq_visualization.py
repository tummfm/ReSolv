import pickle

import matplotlib.pyplot as plt
import pandas as pd


path = 'output/results/'

datedate = {}
loss = {}

datedate[0] = '210927_134629/210927_134629'
datedate[1] = '210917_124429/210917_124429'
datedate[2] = '210914_172156/210914_172156'
datedate[3] = '210914_172138/210914_172138'
datedate[4] = '210917_125210/210917_125210'
datedate[5] = '210917_125138/210917_125138'
datedate[6] = '210923_224015/210923_224015'
datedate[7] = '210923_224029/210923_224029'
datedate[8] = '210923_224055/210923_224055'
datedate[9] = '211006_235525/211006_235525'
datedate[10] = '211006_235551/211006_235551'


# datedate_01 = '210713_133106/210713_133106'
# datedate_02 = '210723_221543/210723_221543'
# datedate_03 = '210809_094902/210809_094902'
# datedate_04 = '210809_094850/210809_094850'
# datedate_05 = '210804_001752/210804_001752'
# datedate_06 = '210804_001705/210804_001705'
# datedate_07 = '210805_222929/210805_222929'
# datedate_08 = '210805_223144/210805_223144'

for ii in range(0,11):
        loss[ii] = {}

        load_path = path + datedate[ii] + '_losses.pkl'
        with open(load_path, 'rb') as f:
                loss[ii]['raw'] = pickle.load(f)
        df_loss = pd.DataFrame(loss[ii])
        df_roll_loss = df_loss.rolling(20).mean()
        loss[ii]['roll'] = df_roll_loss

        # print(loss[ii])


fig, ax = plt.subplots(figsize=(10,10))
ax.set_yscale('log')
typus = 'roll'

ax.plot(loss[0][typus], label='DR 0')
ax.plot(loss[1][typus], label='DR 0.1 output')
ax.plot(loss[2][typus], label='DR 0.2 output')
ax.plot(loss[3][typus], label='DR 0.3 output')
ax.plot(loss[4][typus], label='DR 0.2 output 2x out_embed_size')
ax.plot(loss[5][typus], label='DR 0.3 output 2x out_embed_size')
ax.plot(loss[6][typus], label='DR 0.1 output 2x out_embed_size')
ax.plot(loss[7][typus], label='DR 0.4 output 2x out_embed_size')
ax.plot(loss[8][typus], label='DR 0.4 output')
ax.plot(loss[9][typus], label='DR 0.2 output 4x out_embed_size')
ax.plot(loss[10][typus], label='DR 0.2 output 8x out_embed_size')

# ax.plot(loss[9][typus], label='...')


# ax.set(xlim=[150,200])
# ax.set(ylim=[5*10e-5,2*10e-4])

ax.grid('minor')

fig.legend()

fig.savefig('comparison_training.png')



#############################################################################


path_con = 'output/fuq_convergence/'

num_fuq_samples = '10'

date_01 = '211011_163254'

datedate_01 = date_01 + '/' + date_01
# datedate_01 = '210723_122915/210723_122915'
# datedate_01 = '210725_161921/210725_161921'

meta_rdf_path = path_con + datedate_01 + '_rdf_conv_meta.pkl'
meta_adf_path = path_con + datedate_01 + '_adf_conv_meta.pkl'
meta_do_path = path_con + datedate_01 + '_do_hp_fuq.pkl'


with open(meta_rdf_path, 'rb') as f:
        meta_rdf = pickle.load(f)

with open(meta_adf_path, 'rb') as f:
        meta_adf = pickle.load(f)

with open(meta_do_path, 'rb') as f:
        meta_do = pickle.load(f)

num_conv_samples = len(meta_rdf['values'])

fig, ax = plt.subplots(4,1,figsize=(15,30))
for ii in range(num_conv_samples):
        ax[0].plot(meta_rdf['values'][ii]['mean'])
        ax[0].title.set_text('RDF Mean')

        ax[1].plot(meta_rdf['values'][ii]['std'])
        ax[1].title.set_text('RDF STD')

        ax[2].plot(meta_adf['values'][ii]['mean'])
        ax[2].title.set_text('ADF Mean')

        ax[3].plot(meta_adf['values'][ii]['std'])
        ax[3].title.set_text('ADF STD')

#ax.set(xlim=[0,60], ylim=[0.00,0.015])
for jj in range(4):
        ax[jj].grid('minor')



fig.suptitle('# convergence samples: ' + str(num_conv_samples) + '    dropout rate: ' + str(meta_do['output']['dropout_rate']) + '    # forward samples: ' + str(num_fuq_samples), fontsize=24)

fig.savefig(path_con + date_01 + '/' + date_01 + '_convergence_analysis' + '.png')


print('debug')

#############################################################################

path_con = 'output/fuq_convergence/'

dates = ['210928_114418', '210928_202455', '210928_114359', '210928_202603', '210929_105403'] # 0.0 training
# dates = ['210929_105656', '210929_160523', '210929_160545', '210929_160600', '210930_145651'] # 0.4 training
# dates = ['210930_145608', '210930_002605', '210930_002513', '210930_002544', '210930_145541'] # 0.4 training wide 

samples = [5, 7, 10, 15, 20]
convs = [20, 15, 10, 7, 5]

fig, ax = plt.subplots(4,1,figsize=(10,20))

for ii in range(len(dates)):
        
        date = dates[ii]
        sample = samples[ii]
        conv = convs[ii]

        datedate = date + '/' + date

        meta_score_path = path_con + datedate + '_meta_score_fuq.pkl'
        meta_do_path = path_con + datedate + '_do_hp_fuq.pkl'

        with open(meta_score_path, 'rb') as f:
                meta_score = pickle.load(f)
        
        with open(meta_do_path, 'rb') as f:
                meta_do = pickle.load(f)

        pointlabel = date + ': ' + str(sample) + ' forward samples, ' + str(conv) + ' convergence samples, ' + 'dropout rate = ' + str(meta_do[0]['dropout_rate'])

        ax[0].plot(sample,meta_score['RDF']['mean']['uqint'],'o',label=pointlabel)

        ax[1].plot(sample,meta_score['RDF']['std']['uqint'],'o',label=pointlabel)

        ax[2].plot(sample,meta_score['ADF']['mean']['uqint'],'o',label=pointlabel)

        ax[3].plot(sample,meta_score['ADF']['std']['uqint'],'o',label=pointlabel)


#ax.set(xlim=[0,60], ylim=[0.00,0.015])
for jj in range(4):
        ax[jj].grid('minor')
        ax[jj].set(xlabel='forward samples')        
        if jj == 0:
                # ax[jj].legend()
                ax[jj].legend(loc='lower center', bbox_to_anchor=(0.5, 1.025),
                        fancybox=True, shadow=True)

ax[0].set(ylabel='variance score RDF Mean')        
ax[1].set(ylabel='variance score RDF STD')        
ax[2].set(ylabel='variance score ADF Mean')        
ax[3].set(ylabel='variance score ADF STD')        


fig.suptitle('convergence analysis with different number of samples, \ndropout key = ' + str(42), fontsize=20)

fig.savefig('overview_convergence_analysis' + '.png')

print('debug')


# plot functions

def plot_save_results(loss_history, visible_device, num_updates, start_datetime,
                      rdf_bin_centers, g_average_final, reference_rdf=None,
                      g_average_init=None):
        ## FIGURE
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        color = 'tab:red'
        ax[0].set_xlabel('update step')
        ax[0].set_ylabel('Loss')
        ax[0].plot(loss_history, color=color, label='Loss')
        # ax01 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
        # color = 'tab:blue'
        # ax01.semilogy(gradient_history, label='Gradient norm', color=color)
        # ax01.set_ylabel('Gradient norm', color=color)  # we already handled the x-label with ax1

        ax[1].plot(rdf_bin_centers, g_average_final, label='predicted')
        if reference_rdf is not None:
                ax[1].plot(rdf_bin_centers, reference_rdf, label='reference')
        if g_average_init is not None:
                ax[1].plot(rdf_bin_centers, g_average_init,
                           label='initial guess')

        ax[1].legend()

        plt.savefig('output/results/' + str(start_datetime) + '/' + str(
                start_datetime) + '_figures' + '.png')

        # ## GIF
        # working_dir = 'output/Gif/'
        # with open(working_dir + file_name, 'rb') as f:
        #     plot_dict = pickle.load(f)
        # x_vals = plot_dict['x_vals']
        # reference = plot_dict['reference']
        # time_series = plot_dict['series']

        # fig, ax = plt.subplots(figsize=(5, 3))
        # series_line = ax.plot(x_vals, reference, label='Predicted')[0]
        # ax.plot(x_vals, reference, label='Reference')
        # ax.legend()
        # # ax.set(xlim=(-3, 3), ylim=(-1, 1))

        # def animate(i):
        #     series_line.set_ydata(time_series[i])
        #     ax.set_title('Epoche ' + str(i))

        # file_name = file_name[:-4]
        # anim = FuncAnimation(fig, animate, interval=200, frames=len(time_series) - 1)
        # anim.save(working_dir + file_name + '.gif', writer='imagemagick')

        return


def save_dropout_hyperparams(dropout_hp, filepath):
        with open(filepath, 'wb') as output:
                pickle.dump(dropout_hp, output, pickle.HIGHEST_PROTOCOL)
        return


def plot_fuq_rdf(rdf_meta, rdf_bin_centers, model, visible_device,
                 start_datetime, sample_size, dropout_hp, dropout_key_init,
                 reference_rdf=None):
        plt.figure()
        plt.plot(rdf_bin_centers, rdf_meta['mean'], label='mean', color='red')
        plt.fill_between(rdf_bin_centers,
                         rdf_meta['mean'] + abs(rdf_meta['std']),
                         rdf_meta['mean'] - abs(rdf_meta['std']), label='std',
                         color='orange', alpha=0.2)
        # plt.plot(rdf_bin_centers, rdf_meta['mean']+abs(rdf_meta['std']), label='sigma plus')
        # plt.plot(rdf_bin_centers, rdf_meta['mean']-abs(rdf_meta['std']), label='sigma minus')
        if reference_rdf is not None:
                plt.plot(rdf_bin_centers, reference_rdf, '--', color='blue',
                         label='reference')
        plt.legend()
        plt.grid('minor')
        plt.title('Forward UQ RDF:' + ' DR = ' + str(
                dropout_hp['output']['dropout_rate']) + '; Samples = ' + str(
                sample_size) + '; init Key = ' + str(dropout_key_init))
        plt.savefig('output/fuq/' + str(start_datetime) + '/' + str(
                start_datetime) + '_rdf_figure' + '.png')
        plt.savefig(
                'Figures/Forward_UQ_' + model + str(visible_device) + '.png')
        return


def plot_fuq_adf(adf_meta, adf_bin_centers, model, visible_device,
                 start_datetime, sample_size, dropout_hp, dropout_key_init,
                 reference_adf=None):
        plt.figure()
        plt.plot(adf_bin_centers, adf_meta['mean'], label='mean', color='red')
        plt.fill_between(adf_bin_centers,
                         adf_meta['mean'] + abs(adf_meta['std']),
                         adf_meta['mean'] - abs(adf_meta['std']), label='std',
                         color='orange', alpha=0.2)

        if reference_adf is not None:
                plt.plot(adf_bin_centers, reference_adf, '--', color='blue',
                         label='reference')
        plt.legend()
        plt.grid('minor')
        plt.title('Forward UQ ADF:' + ' DR = ' + str(
                dropout_hp['output']['dropout_rate']) + '; Samples = ' + str(
                sample_size) + '; init Key = ' + str(dropout_key_init))
        plt.savefig('output/fuq/' + str(start_datetime) + '/' + str(
                start_datetime) + '_adf_figure' + '.png')
        plt.savefig(
                'Figures/Forward_UQ_' + model + str(visible_device) + '.png')
        return


def plot_params(params, block, name):
        layer = {}

        layer[0] = 'Upprojection'
        layer[1] = 'Dense_Series'
        layer[2] = 'Dense_Series_1'
        layer[3] = 'Dense_Series_2'
        layer[4] = 'Final_output'

        block = block

        pref = 'Energy/~/' + block + '/~/'

        nlayers = len(layer)

        fig, axs = plt.subplots(ncols=2, nrows=nlayers)
        fig.set_figwidth(12)
        fig.set_figheight(12)
        fig.suptitle(name)

        cmap = sns.color_palette("deep", 3)

        for row, module in enumerate(sorted(params)):
                for ii in range(nlayers):
                        path = pref + layer[ii]
                        if module == path:
                                # for r, mod in enumerate(sorted(params[module])):

                                ax = axs[ii][0]
                                sns.heatmap(params[path]["w"], cmap=cmap, ax=ax,
                                            vmin=-1e-7, vmax=1e-7)

                                colorbar = ax.collections[0].colorbar
                                colorbar.set_ticks([-1e-7, 0, 1e-7])
                                colorbar.set_ticklabels(['< 0', '= 0', '> 0'])

                                ax.title.set_text(f"{path}/w")

                                if (layer[ii] is not 'Upprojection') and (
                                        layer[ii] is not 'Final_output'):
                                        ax = axs[ii][1]
                                        b = np.expand_dims(params[path]["b"],
                                                           axis=0)
                                        sns.heatmap(b, cmap=cmap, ax=ax,
                                                    vmin=-1e-7, vmax=1e-7)

                                        colorbar = ax.collections[0].colorbar
                                        colorbar.set_ticks([-1e-7, 0, 1e-7])
                                        colorbar.set_ticklabels(
                                                ['< 0', '= 0', '> 0'])

                                        ax.title.set_text(f"{path}/b")

                                fig.tight_layout()
                                plt.savefig(
                                        'params_plot_' + block + '_' + name + '.png')


def plot_outputs(output_raw, output, ilayer, name):
        fig, axs = plt.subplots(ncols=2, nrows=1)
        fig.set_figwidth(12)
        fig.set_figheight(12)
        fig.suptitle(name + '_' + str(ilayer))

        cmap = sns.color_palette("deep", 3)

        ax = axs[0]
        sns.heatmap(output_raw, cmap=cmap, ax=ax, vmin=-1e-7, vmax=1e-7)

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([-1e-7, 0, 1e-7])
        colorbar.set_ticklabels(['< 0', '= 0', '> 0'])

        ax.title.set_text('output raw')

        ax = axs[1]
        sns.heatmap(output, cmap=cmap, ax=ax, vmin=-1e-7, vmax=1e-7)

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([-1e-7, 0, 1e-7])
        colorbar.set_ticklabels(['< 0', '= 0', '> 0'])

        ax.title.set_text('output droped out')

        fig.tight_layout()
        plt.savefig('netoutput_' + name + '_' + str(ilayer) + '.png')


def fuq_score(rdf_dict, adf_dict):
        conv_result_dict = {}

        dict_list = [rdf_dict, adf_dict]
        num_targets = len(dict_list)

        for ii in range(num_targets):

                dict = dict_list[ii]

                means = []
                stds = []

                for key in dict['values'].keys():
                        means.append(dict['values'][key]['mean'])
                        stds.append(dict['values'][key]['std'])

                means = onp.asarray(means)
                stds = onp.asarray(stds)

                if ii == 0:
                        target = 'RDF'

                elif ii == 1:
                        target = 'ADF'

                conv_result_dict[target] = {}

                conv_result_dict[target]['mean'] = {}
                conv_result_dict[target]['mean']['std'] = means.std(axis=0)
                conv_result_dict[target]['mean']['uqint'] = onp.square(
                        conv_result_dict[target]['mean']['std']).sum()

                conv_result_dict[target]['std'] = {}
                conv_result_dict[target]['std']['std'] = stds.std(axis=0)
                conv_result_dict[target]['std']['uqint'] = onp.square(
                        conv_result_dict[target]['std']['std']).sum()

        return conv_result_dict
