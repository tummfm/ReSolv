from jax import vmap
import jax.numpy as jnp
from chemtrain.sparse_graph import angle
from jax_md import space
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as clr
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
import aa_simulations.aa_util as aa

# def ax_positions():
#     ax_positions = [axes[0,0].get_position(), axes[0,1].get_position(), axes[1,0].get_position(), axes[1,1].get_position()]
#     with open('output/saved_optimization_results/Publication_results/axis_positions.pkl', 'wb') as f:
#         pickle.dump(ax_positions, f)


def plot_rdf_adf_tcf(bin_centers_list, g_average_list, model, reference_list=None,
                                   g_average_init_list=None, transparent=False, labels=None):
    size_x = 6.4 * 1.5
    size_y = 4.8 * 1.5
    left_shift = -0.

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=[size_x, size_y], constrained_layout=False)


    if labels is None:
        labels = ['reference','predicted','initial guess']
    
    # RDF in ax1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel('r in $\mathrm{nm}$')
    ax1.set_ylabel('RDF')
    ax1.plot(bin_centers_list[0], g_average_list[0], color='#00A087FF')
    # plt.plot(bin_centers, g_average_final, label=labels[1], color='#91D1C2FF')   
    if g_average_init_list is not None:
        ax1.plot(bin_centers_list[0], g_average_init_list[0], color='#3C5488FF')
        # plt.plot(bin_centers, g_average_init, label=labels[2], color='#8491B4FF')
    if reference_list is not None:
        ax1.plot(bin_centers_list[0], reference_list[0], dashes=(4, 3),
                                                color='k', linestyle='--')    

    # ax1.legend(loc="upper right")
    ax1.text(-0.2, 1., '$\mathbf{a}$', transform=ax1.transAxes, fontsize=12)
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)
    
    # predicted and reference TCF in ax2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(bin_centers_list[2], g_average_list[2], color='#00A087FF')  
    if g_average_init_list is not None:
        ax2.plot(bin_centers_list[2], g_average_init_list[2], color='#3C5488FF')
    if reference_list is not None:
        ax2.plot(bin_centers_list[2], reference_list[2],  dashes=(4, 3),
                                                color='k', linestyle='--')
    ax2.set_xlabel('r in $\mathrm{nm}$')
    ax2.set_ylabel('TCF')
    ax2.text(-0.2, 1., '$\mathbf{c}$', transform=ax2.transAxes, fontsize=12)
    # ax_position_2 = ax_positions[0]
    # ax_position_2 = [ax_position_2.x0 - left_shift, ax_position_2.y0,  ax_position_2.width, ax_position_2.height]
    # ax2.set_position(ax_position_2)

    # predicted and reference ADF in ax3
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(bin_centers_list[1], g_average_list[1], label=labels[1], color='#00A087FF')
    if g_average_init_list is not None:
        ax3.plot(bin_centers_list[1], g_average_init_list[1], label=labels[2], color='#3C5488FF')
    if reference_list is not None:
        ax3.plot(bin_centers_list[1], reference_list[1], label=labels[0], dashes=(4, 3),
                                                color='k', linestyle='--')
    ax3.set_xlabel(r'$\alpha$ in $\mathrm{rad}$')
    ax3.set_ylabel('ADF')
    # ax3.set_ylim([-1, 2])
    # ax3.set_ylim([-1, 2])
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    #specify order of items in legend
    order = [2,0,1]
    ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=(0, -0.455)) #-0.76
    # ax3.legend(loc=(0, -0.76))
    ax3.text(-0.175, 1., '$\mathbf{b}$', transform=ax3.transAxes, fontsize=12)
    # ax_position_3 = ax_positions[1]
    # ax_position_3 = [ax_position_3.x0 - left_shift, ax_position_3.y0, ax_position_3.width, ax_position_3.height]
    # ax3.set_position(ax_position_3)

    if transparent:
        plt.savefig('output/figures/predicted_' + model + '.pdf', transparent=True)
    else:
        plt.savefig('output/figures/predicted_' + model + '.pdf')
    return


def dihedral_map():
    mymap = onp.array([[0.9, 0.9, 0.9],
                       [0.85, 0.85, 0.85],
                       [0.8 ,0.8, 0.8],
                       [0.75, 0.75, 0.75],
                       [0.7 ,0.7, 0.7],
                       [0.65, 0.65, 0.65],
                       [0.6 ,0.6, 0.6],
                       [0.55, 0.55, 0.55],
                       [0.5 ,0.5, 0.5],
                       [0.45, 0.45, 0.45],
                       [0.4 ,0.4, 0.4],
                       [0.35, 0.35, 0.35],
                       [0.3 ,0.3, 0.3],
                       [0.25, 0.25, 0.25],
                       [0.2 ,0.2, 0.2],
                       [0.15, 0.15, 0.15],
                       [0.1 ,0.1, 0.1],
                       [0.05, 0.05, 0.05],
                       [0, 0, 0]])

    newcmp = clr.ListedColormap(mymap)
    return newcmp


def plot_histogram_density_1x3(list_angles,saveas,titles=None,folder=''):
    '''Plot and save 2D histogram for alanine dipeptide density
    from the dihedral angles.'''
    newcmp = dihedral_map()
    numbering = ['$\mathbf{a}$','$\mathbf{b}$','$\mathbf{c}$']
    n_plots = len(list_angles)

    scale_y = 1.0
    scale_x = 3
        
    # Create 1x3 sub plots
    fig, axs= plt.subplots(ncols=n_plots, figsize=[6.4 * scale_x, 4.8 * scale_y], constrained_layout=True)

    # fig, axs = plt.subplots(ncols=n_plots, figsize=(6.4*scale_x,4.8),constrained_layout=True)
    
    images = []
    for i in range(n_plots):
        h, x_edges, y_edges  = jnp.histogram2d(list_angles[i][:,0],
                            list_angles[i][:,1], bins = 60, density=True)
        h_masked = onp.where(h == 0, onp.nan, h)
        x, y = onp.meshgrid(x_edges, y_edges)
        images.append(axs[i].pcolormesh(x,y,h_masked.T, cmap=newcmp))
        axs[i].text(-0.2, 1., numbering[i], transform=axs[i].transAxes, fontsize=24)
        axs[i].set_xlim([-180,180])
        axs[i].set_ylim([-180,180])
        axs[i].set_xlabel('$\phi$ in $\mathrm{deg}$')
        axs[i].set_ylabel('$\psi$ in $\mathrm{deg}$')
        if titles:
            axs[i].set_title(titles[i])
        axs[i].text(-155,90,'$C5$',fontsize=18)
        axs[i].text(-70,90,'$C7eq$',fontsize=18)
        axs[i].text(145,90,'$C5$',fontsize=18)
        axs[i].text(-155,-150,'$C5$',fontsize=18)
        axs[i].text(-70,-150,'$C7eq$',fontsize=18)
        axs[i].text(145,-150,'$C5$',fontsize=18)
        axs[i].text(-170,-90,r'$\alpha_R$"',fontsize=18)
        axs[i].text(140,-90,r'$\alpha_R$"',fontsize=18)
        axs[i].text(-70,-90,r'$\alpha_R$',fontsize=18)
        axs[i].text(70,0,r'$\alpha_L$',fontsize=18)
        axs[i].plot([-180,13],[74,74],'k',linewidth=0.5)
        axs[i].plot([128,180],[74,74],'k',linewidth=0.5)
        axs[i].plot([13,13],[-180,180],'k',linewidth=0.5)
        axs[i].plot([128,128],[-180,180],'k',linewidth=0.5)
        axs[i].plot([-180,13],[-125,-125],'k',linewidth=0.5)
        axs[i].plot([128,180],[-125,-125],'k',linewidth=0.5)
        axs[i].plot([-134,-134],[-125,74],'k',linewidth=0.5)
        axs[i].plot([-110,-110],[-180,-125],'k',linewidth=0.5)
        axs[i].plot([-110,-110],[74,180],'k',linewidth=0.5)

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Density')
    
    # def update(changed_image):
    #     for im in images:
    #         if (changed_image.get_cmap() != im.get_cmap()
    #                 or changed_image.get_clim() != im.get_clim()):
    #             im.set_cmap(changed_image.get_cmap())
    #             im.set_clim(changed_image.get_clim())


    # for im in images:
    #     im.callbacks.connect('changed', update)
    plt.savefig(f'plots/postprocessing/{folder}histogram_compare_density_{saveas}.pdf')
    plt.show()
    plt.close('all')
    return


def plot_histogram_free_energy_1x3(list_angles,saveas,kbT=2.49435321,degrees=True,titles=None,folder=''):
    '''Plot and save 2D histogram for alanine dipeptide free energies
    from the dihedral angles.'''
    cmap = plt.get_cmap('magma')
    numbering = ['$\mathbf{a}$','$\mathbf{b}$','$\mathbf{c}$']
    n_plots = len(list_angles)

    scale_y = 1.0
    scale_x = 3

    # Create 1x3 sub plots
    fig, axs= plt.subplots(ncols=n_plots, figsize=[6.4 * scale_x, 4.8 * scale_y], constrained_layout=True)

    # fig, axs = plt.subplots(ncols=n_plots, figsize=(6.4*scale_x,4.8),constrained_layout=True)
    
    images = []
    for i in range(n_plots):
        if degrees:
            list_angles[i] = jnp.deg2rad(list_angles[i])
        h, x_edges, y_edges  = jnp.histogram2d(list_angles[i][:,0],
                            list_angles[i][:,1], bins = 60, density=True)
        h_masked = onp.log(h)*-kbT/4.184
        x, y = onp.meshgrid(x_edges, y_edges)
        images.append(axs[i].pcolormesh(x,y,h_masked.T, cmap=cmap))
        axs[i].text(-0.2, 1., numbering[i], transform=axs[i].transAxes, fontsize=24)
        axs[i].set_xlim([-onp.pi,onp.pi])
        axs[i].set_ylim([-onp.pi,onp.pi])
        axs[i].set_xlabel('$\phi$ in $\mathrm{rad}$')
        axs[i].set_ylabel('$\psi$ in $\mathrm{rad}$')
        if titles:
            axs[i].set_title(titles[i])

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Free Energy (kcal/mol)')
    
    # def update(changed_image):
    #     for im in images:
    #         if (changed_image.get_cmap() != im.get_cmap()
    #                 or changed_image.get_clim() != im.get_clim()):
    #             im.set_cmap(changed_image.get_cmap())
    #             im.set_clim(changed_image.get_clim())


    # for im in images:
    #     im.callbacks.connect('changed', update)
    plt.savefig(f'plots/postprocessing/{folder}histogram_compare_free_energy_{saveas}.pdf')
    plt.show()
    plt.close('all')
    return


def plot_histogram_density_2x2(list_angles,saveas,titles=None,folder=''):
    '''Plot and save 2D histogram for alanine dipeptide free energies
    from the dihedral angles.'''
    newcmp = dihedral_map()
    numbering = ['$\mathbf{a}$','$\mathbf{b}$','$\mathbf{c}$']
    n_plots = len(list_angles)
    scale_x = 1.5
    scale_y = 1.5
    left_shift = -0.

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=[6.4*scale_x, 4.8*scale_y], constrained_layout=False)
        
    # Create 1x3 sub plots
    # fig, axs= plt.subplots(ncols=n_plots, figsize=[6.4 * scale_x, 4.8 * scale_y], constrained_layout=True)
    axs = []
    images = []
    for i in range(n_plots):
        ax_temp = fig.add_subplot(gs[i//2, i%2])
        axs.append(ax_temp)

        h, x_edges, y_edges  = jnp.histogram2d(list_angles[i][:,0],
                            list_angles[i][:,1], bins = 60, density=True)
        h_masked = onp.where(h == 0, onp.nan, h)
        x, y = onp.meshgrid(x_edges, y_edges)
        images.append(axs[i].pcolormesh(x,y,h_masked.T, cmap=newcmp))
        if i%2==1:
            axs[i].text(-0.175, 1., numbering[i], transform=axs[i].transAxes, fontsize=12)
        else:
            axs[i].text(-0.2, 1., numbering[i], transform=axs[i].transAxes, fontsize=12)
        axs[i].set_xlim([-180,180])
        axs[i].set_ylim([-180,180])
        axs[i].set_xlabel('$\phi$ in $\mathrm{deg}$')
        axs[i].set_ylabel('$\psi$ in $\mathrm{deg}$')
        if titles:
            axs[i].set_title(titles[i])
        axs[i].text(-155,90,'$C5$',fontsize=12)
        axs[i].text(-70,90,'$C7eq$',fontsize=12)
        axs[i].text(145,90,'$C5$',fontsize=12)
        axs[i].text(-155,-150,'$C5$',fontsize=12)
        axs[i].text(-70,-150,'$C7eq$',fontsize=12)
        axs[i].text(145,-150,'$C5$',fontsize=12)
        axs[i].text(-170,-90,r'$\alpha_R$"',fontsize=12)
        axs[i].text(140,-90,r'$\alpha_R$"',fontsize=12)
        axs[i].text(-70,-90,r'$\alpha_R$',fontsize=12)
        axs[i].text(70,0,r'$\alpha_L$',fontsize=12)
        axs[i].plot([-180,13],[74,74],'k',linewidth=0.5)
        axs[i].plot([128,180],[74,74],'k',linewidth=0.5)
        axs[i].plot([13,13],[-180,180],'k',linewidth=0.5)
        axs[i].plot([128,128],[-180,180],'k',linewidth=0.5)
        axs[i].plot([-180,13],[-125,-125],'k',linewidth=0.5)
        axs[i].plot([128,180],[-125,-125],'k',linewidth=0.5)
        axs[i].plot([-134,-134],[-125,74],'k',linewidth=0.5)
        axs[i].plot([-110,-110],[-180,-125],'k',linewidth=0.5)
        axs[i].plot([-110,-110],[74,180],'k',linewidth=0.5)

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    cbaxes = fig.add_subplot(gs[1, 1])
    bb = cbaxes.get_position()
    bb.x0 = 0.55
    cbaxes.set_position(bb)

    bb = cbaxes.get_position()
    bb.x1 = 0.57
    cbaxes.set_position(bb)

    cbar = fig.colorbar(images[0], cax=cbaxes)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Density')
    
    # def update(changed_image):
    #     for im in images:
    #         if (changed_image.get_cmap() != im.get_cmap()
    #                 or changed_image.get_clim() != im.get_clim()):
    #             im.set_cmap(changed_image.get_cmap())
    #             im.set_clim(changed_image.get_clim())


    # for im in images:
    #     im.callbacks.connect('changed', update)
    plt.savefig(f'plots/postprocessing/{folder}histogram_density_2x2_{saveas}.pdf')
    plt.show()
    plt.close('all')
    return


def plot_compare_scatter_forces(predicted_list,reference,save_as,model_list,line=2000,
                                                                            max=55000):
    '''Scatter plot and save predicted and reference forces'''

    size_x = 6.4 * 1.5
    size_y = 4.8 * 1.5
    left_shift = -0.
    scale_y = 0.9
    # Create 2x2 sub plots

    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=[size_x, 4.8 * scale_y], constrained_layout=True)

    # FM in ax1
    ax1.set_xlabel('Predicted Force Components in kJ $\mathrm{mol^{-1} \ nm^{-1}}$')
    ax1.set_ylabel('Reference Force Components in kJ $\mathrm{mol^{-1} \ nm^{-1}}$')
    ax1.set_title(model_list[0])
    ax1.hexbin(predicted_list[0],reference, gridsize=50, mincnt=1, vmax=max)
    ax1.plot([-line,line],[-line,line],'r--')
    # ax1.legend(loc="upper right")
    ax1.text(-0.2, 1., '$\mathbf{a}$', transform=ax1.transAxes, fontsize=12)
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)

    # RE in ax2
    ax2.set_xlabel('Predicted Force Components in kJ $\mathrm{mol^{-1} \ nm^{-1}}$')
    ax2.set_ylabel('Reference Force Components in kJ $\mathrm{mol^{-1} \ nm^{-1}}$')
    ax2.set_title(model_list[1])
    hex = ax2.hexbin(predicted_list[0],reference, gridsize=50, mincnt=1, vmax=max)
    ax2.plot([-line,line],[-line,line],'r--')
    # ax1.legend(loc="upper right")
    ax2.text(-0.2, 1., '$\mathbf{b}$', transform=ax2.transAxes, fontsize=12)
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)

    cbar = fig.colorbar(hex, ax=ax2)
    #Print 100000 as 100K
    cbar.ax.set_yticklabels(['{:,.0f}K'.format(i/1000) for i in cbar.get_ticks()])
    cbar.set_label('Number of data points')
    plt.savefig('plots/postprocessing/force_scatter_compare_'+save_as+'.pdf')
    plt.show()
    return


def plot_timestep_4x4(x_times, mean_list, std_list, save_as, labels=None):
    size_x = 6.4 * 1.5
    size_y = 4.8 * 1.5
    left_shift = -0.

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=[size_x, size_y], constrained_layout=False)


    if labels is None:
        labels = ['FM','RE']
    
    # RDF in ax1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel('$\Delta t$ in fs')
    ax1.set_ylabel('RDF MSE')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax1.errorbar(x_times, mean_list[0][:,0], std_list[0][:,0], marker='o', color='#00A087FF')
    ax1.errorbar(x_times, mean_list[1][:,0], std_list[1][:,0], marker='o', color='#3C5488FF')
    

    # ax1.legend(loc="upper right")
    ax1.text(-0.2, 1., '$\mathbf{a}$', transform=ax1.transAxes, fontsize=12)
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)
    
    # predicted and reference TCF in ax2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlabel('$\Delta t$ in fs')
    ax2.set_ylabel('TCF MSE')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax2.errorbar(x_times, mean_list[0][:,2], std_list[0][:,2], marker='o', color='#00A087FF')
    ax2.errorbar(x_times, mean_list[1][:,2], std_list[1][:,2], marker='o', color='#3C5488FF')
    

    # ax1.legend(loc="upper right")
    ax2.text(-0.2, 1., '$\mathbf{c}$', transform=ax2.transAxes, fontsize=12)
    # ax_position_2 = ax_positions[0]
    # ax_position_2 = [ax_position_2.x0 - left_shift, ax_position_2.y0,  ax_position_2.width, ax_position_2.height]
    # ax2.set_position(ax_position_2)

    # predicted and reference TCF in ax3
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax3.errorbar(x_times, mean_list[0][:,1], std_list[0][:,1], marker='o', color='#00A087FF',
                                                                    label=labels[0])
    ax3.errorbar(x_times, mean_list[1][:,1], std_list[1][:,1], marker='o', color='#3C5488FF',
                                                                    label=labels[1])
    ax3.set_xlabel('$\Delta t$ in fs')
    ax3.set_ylabel('ADF MSE')
    ax3.legend(loc=(0, -0.37)) #-0.76
    # ax3.legend(loc=(0, -0.76))
    ax3.text(-0.175, 1., '$\mathbf{b}$', transform=ax3.transAxes, fontsize=12)
    # ax_position_3 = ax_positions[1]
    # ax_position_3 = [ax_position_3.x0 - left_shift, ax_position_3.y0, ax_position_3.width, ax_position_3.height]
    # ax3.set_position(ax_position_3)

    plt.savefig('output/figures/Timestep_'+save_as+'.pdf')
    return


def plot_timestep_1x3(x_times, mean_list, std_list, save_as, labels=None, extra=None,
                        points=None):
    left_shift = -0.
    scale_y = 0.75
    scale_x = 1.5
    
    if labels is None:
        labels = ['FM','RE']
    
    # Create 1x3 sub plots
    fig, (ax1, ax2, ax3)= plt.subplots(1, 3, figsize=[6.4 * scale_x, 4.8 * scale_y], constrained_layout=True)

    # RDF in ax1
    ax1.set_xlabel('$\Delta t_\mathrm{CG}$ in fs')
    ax1.set_ylabel('RDF MSE')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax1.errorbar(x_times, mean_list[0][:,0], std_list[0][:,0], marker='o', color='#00A087FF')#,
                                                                    # label=labels[0])
    ax1.errorbar(x_times, mean_list[1][:,0], std_list[1][:,0], marker='o', color='#3C5488FF')#,
                                                                    # label=labels[1])
    if points is not None:
        n_points_FM = points[0].shape[0]
        n_points_RE = points[1].shape[0]
        for p in range(n_points_FM):
            ax1.scatter(x_times,points[0][p,:,0], color='#91D1C2FF')
        for q in range(n_points_RE):
            ax1.scatter(x_times,points[1][q,:,0], color='#8491B4FF')
    if extra is not None:
        # ax1.plot(x_times[0],extra[0], marker='^', color='#4DBBD5FF')
        ax1.errorbar(x_times[0], extra[0][0], extra[1][0], marker='^', color='#4DBBD5FF')
    # ax1.legend(loc=(0, -0.37))
    ax1.text(-0.2, 1., '$\mathbf{a}$', transform=ax1.transAxes, fontsize=12)
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)

    # ADF in ax2
    ax2.set_xlabel('$\Delta t_\mathrm{CG}$ in fs')
    ax2.set_ylabel('ADF MSE')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    # ax2.locator_params(axis='y', nbins=8)
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.set_yticks([0,0.00005,0.0001,0.00015])
    ax2.errorbar(x_times, mean_list[0][:,1], std_list[0][:,1], marker='o', color='#00A087FF')
    ax2.errorbar(x_times, mean_list[1][:,1], std_list[1][:,1], marker='o', color='#3C5488FF')
    if points is not None:
        n_points_FM = points[0].shape[0]
        n_points_RE = points[1].shape[0]
        for p in range(n_points_FM):
            ax2.scatter(x_times,points[0][p,:,1], color='#91D1C2FF')
        for q in range(n_points_RE):
            ax2.scatter(x_times,points[1][q,:,1], color='#8491B4FF')
    if extra is not None:
        # ax2.plot(x_times[0],extra[1], marker='^', color='#4DBBD5FF')
        ax2.errorbar(x_times[0], extra[0][1], extra[1][1], marker='^', color='#4DBBD5FF')
    # ax1.legend(loc="upper right")
    ax2.text(-0.2, 1., '$\mathbf{b}$', transform=ax2.transAxes, fontsize=12)
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)

    ax3.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    ax3.errorbar(x_times, mean_list[0][:,2], std_list[0][:,2], marker='o', color='#00A087FF',
                                                                    label=labels[0])
    ax3.errorbar(x_times, mean_list[1][:,2], std_list[1][:,2], marker='o', color='#3C5488FF',
                                                                    label=labels[1])
    if points is not None:
        n_points_FM = points[0].shape[0]
        n_points_RE = points[1].shape[0]
        for p in range(n_points_FM):
            ax3.scatter(x_times,points[0][p,:,2], color='#91D1C2FF')
        for q in range(n_points_RE):
            ax3.scatter(x_times,points[1][q,:,2], color='#8491B4FF')
    if extra is not None:
        # ax3.plot(x_times[0],extra[2], marker='^', color='#4DBBD5FF')
        ax3.errorbar(x_times[0], extra[0][2], extra[1][2], label='RE\n10 fs', marker='^', color='#4DBBD5FF')
    ax3.set_xlabel('$\Delta t_\mathrm{CG}$ in fs')
    ax3.set_ylabel('TCF MSE')
    # ax3.legend(loc=(0, -0.37)) #-0.76
    # ax3.legend(loc=(0, -0.76))
    ax3.legend(loc='upper left', bbox_to_anchor=(1.0,1.025))
    ax3.text(-0.175, 1., '$\mathbf{c}$', transform=ax3.transAxes, fontsize=12)
    # ax_position_3 = ax_positions[1]
    # ax_position_3 = [ax_position_3.x0 - left_shift, ax_position_3.y0, ax_position_3.width, ax_position_3.height]
    # ax3.set_position(ax_position_3)
 
    plt.savefig('output/figures/Timestep_1x3_'+save_as+'.pdf')
    plt.show()
    return


def plot_1D_dihedral_1x2(angles_phi, angles_psi, saveas, labels, bins=60, degrees=True,
                        color=None, line=None):
    '''Plot and save 1D histogram spline for alanine dipeptide dihedral
    angles with mean and standard deviation.
    angles: angles in form of list of [Ntrajectory x Nangles] or
    numpy arrays of [Nmodels, Ntrajectory, Nangles].
    Paper plots in 1x2.'''

    left_shift = -0.
    scale_x = 2.0
    scale_y = 0.9

    # Create 1x2 sub plots
    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=[6.4 * scale_x, 4.8 * scale_y],
                                                        constrained_layout=True)
    if labels is None:
        labels = ['Reference','FM','RE']
    if color is None:
        color = ['#00A087FF','#3C5488FF','k']
    if line is None:
        line = ['-','-','--']
    n_models = len(angles_phi)
    
    for i in range(n_models):
        if degrees:
            angles_conv_phi = angles_phi[i]
            angles_conv_psi = angles_psi[i]
        else:
            angles_conv_phi = onp.rad2deg(angles_phi[i])
            angles_conv_psi = onp.rad2deg(angles_psi[i])
        n_traj = angles_conv_phi.shape[0]
        h_temp = onp.zeros((n_traj,bins,2))
        for j in range(n_traj):
            h_phi, x_bins  = jnp.histogram(angles_conv_phi[j,:], bins=bins,
                                                        density=True)
            h_psi, _  = jnp.histogram(angles_conv_psi[j,:], bins=bins,
                                                        density=True)
            width = x_bins[1]-x_bins[0]
            bin_center = x_bins + width/2
            h_temp[j,:,0] = h_phi
            h_temp[j,:,1] = h_psi
        h_mean = jnp.mean(h_temp, axis=0)
        h_std = jnp.std(h_temp, axis=0)

        ax1.plot(bin_center[:-1], h_mean[:,0], label=labels[i], color=color[i],
                                        linestyle=line[i], linewidth=2.0)
        ax2.plot(bin_center[:-1], h_mean[:,1], label=labels[i], color=color[i],
                                        linestyle=line[i], linewidth=2.0)
        ax1.fill_between(bin_center[:-1], h_mean[:,0]-h_std[:,0], h_mean[:,0]+h_std[:,0],
                                                color=color[i],alpha=0.4)
        ax2.fill_between(bin_center[:-1], h_mean[:,1]-h_std[:,1], h_mean[:,1]+h_std[:,1],
                                                color=color[i],alpha=0.4)
    ax1.set_xlabel('$\phi$ in deg')
    ax1.set_ylabel('Density')
    ax1.text(-0.2, 1., '$\mathbf{a}$', transform=ax1.transAxes, fontsize=16)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)

    ax2.set_xlabel('$\psi$ in deg')
    ax2.set_ylabel('Density')
    ax2.text(-0.2, 1., '$\mathbf{b}$', transform=ax2.transAxes, fontsize=16)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
    # ax_position_1 = ax_positions[2]
    # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
    # ax1.set_position(ax_position_1)

    #get handles and labels
    handles, lbs = plt.gca().get_legend_handles_labels()
    #specify order of items in legend
    n_labels = len(labels)
    order = onp.arange(n_labels, dtype=int)
    #Reference to plot first
    order -= 1
    order[0] = n_labels-1
    ax2.legend([handles[idx] for idx in order],[lbs[idx] for idx in order],
                                loc='upper left', bbox_to_anchor=(1.0,1.025),
                                fontsize=11)
    plt.savefig(f'plots/postprocessing/dihedral_1D_1x2_{saveas}.pdf')
    plt.show()
    plt.close('all')
    return


def plot_pretrain_1x2(displacement, saveas, update_name, nbins=60, colors=None, split=40):
    psi_indices, phi_indices = [3, 4, 6, 8], [1, 3, 4, 6]

    n_models = len(update_name)
    mean_ref = onp.zeros((7,nbins,2))

    #Random seed atomistic reference simulations
    phi_angles_random = onp.load('alanine_dipeptide/confs/phi_angles_r100ns.npy')
    psi_angles_random = onp.load('alanine_dipeptide/confs/psi_angles_r100ns.npy')

    if colors is None:
        colors = pl.cm.Blues(onp.linspace(0.15,1,n_models+1))

    left_shift = -0.
    scale_x = 2.0
    scale_y = 0.9

    # Create 1x2 sub plots
    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=[6.4 * scale_x, 4.8 * scale_y],
                                                        constrained_layout=True)

    for k in range(n_models):
        positions_pretrain = onp.load('../examples/output/alanine_confs/'+
                        f'confs_alanine_RE_{update_name[k]}up_pretrain_all.npy')
        
        phi_angle_pretrain = vmap(aa.one_dihedral_displacement, (0,None,None))(positions_pretrain,
                                                        displacement, phi_indices)
        psi_angle_pretrain = vmap(aa.one_dihedral_displacement, (0,None,None))(positions_pretrain,
                                                            displacement, psi_indices)



        phi_split = phi_angle_pretrain.reshape((split,-1))
        h_temp_phi = onp.zeros((split,nbins))
        psi_split = psi_angle_pretrain.reshape((split,-1))
        h_temp_psi = onp.zeros((split,nbins))

        for m in range(split):
            h_phi, x_bins  = jnp.histogram(phi_split[m,:], bins=nbins,
                                                        density=True)
            h_psi, _  = jnp.histogram(psi_split[m,:], bins=nbins,
                                                        density=True)
            width = x_bins[1]-x_bins[0]
            bin_center = x_bins + width/2
            h_temp_phi[m] = h_phi
            h_temp_psi[m] = h_psi

        h_mean_phi = jnp.mean(h_temp_phi, axis=0)
        h_mean_psi = jnp.mean(h_temp_psi, axis=0)

        if k < 7:
            h_random_phi, _  = onp.histogram(phi_angles_random[k,:], bins=nbins,
                                                            density=True)
            h_random_psi, _  = onp.histogram(psi_angles_random[k,:], bins=nbins,
                                                            density=True)
            mean_ref[k,:,0] = h_random_phi
            mean_ref[k,:,1] = h_random_psi
        
        if k == 0:
            ax1.plot(bin_center[:-1], h_mean_phi, color='#00A087FF', linewidth=2.0)
            ax2.plot(bin_center[:-1], h_mean_psi, label='FM', color='#00A087FF', linewidth=2.0)
        else:
            ax1.plot(bin_center[:-1], h_mean_phi, color=colors[k], linewidth=2.0)
            ax2.plot(bin_center[:-1], h_mean_psi, label='FM + RE ('+
                              update_name[k]+')', color=colors[k], linewidth=2.0)
             # FM in ax1

        ax1.set_xlabel('$\phi$ in deg')
        ax1.set_ylabel('Density')
        ax1.text(-0.2, 1., '$\mathbf{a}$', transform=ax1.transAxes, fontsize=16)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
        # ax_position_1 = ax_positions[2]
        # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
        # ax1.set_position(ax_position_1)

        ax2.set_xlabel('$\psi$ in deg')
        ax2.set_ylabel('Density')
        ax2.text(-0.2, 1., '$\mathbf{b}$', transform=ax2.transAxes, fontsize=16)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,0))
        # ax_position_1 = ax_positions[2]
        # ax_position_1 = [ax_position_1.x0 - left_shift, ax_position_1.y0,  ax_position_1.width, ax_position_1.height]
        # ax1.set_position(ax_position_1)


    mean_phi = onp.mean(mean_ref,axis=0)
    mean_psi = onp.mean(mean_ref,axis=0)

    ax1.plot(bin_center[:-1], mean_phi[:,0], color='k',
                            dashes=(4, 3), linestyle='--', linewidth=2.0)
    ax2.plot(bin_center[:-1], mean_psi[:,1], label='Reference', color='k',
                            dashes=(4, 3), linestyle='--', linewidth=2.0)

    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    #specify order of items in legend
    n_labels = len(labels)
    order = onp.arange(n_labels, dtype=int)
    #Reference to plot first
    order -= 1
    order[0] = n_labels-1
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                                loc='upper left', bbox_to_anchor=(1.0,1.025),
                                fontsize=11)
    plt.savefig(f'plots/postprocessing/pretrain_1x2_{saveas}.pdf')
    plt.show()
    plt.close('all')
    return
    

