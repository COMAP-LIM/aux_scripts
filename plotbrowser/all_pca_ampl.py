import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import numpy.fft as fft
import matplotlib.ticker as mticker

try:
    filename = sys.argv[1]
except IndexError:
    print('Missing filename!')
    print('Usage: python plot_pca_ampl.py filename')
    sys.exit(1)

# folder = '' #/mn/stornext/d16/www_cmb/comap/pca_ampl/'
#folder = '/mn/stornext/d16/cmbco/comap/protodir/plot_browser/'
folder = 'test_figs/'
for k in range(2, 20):
    filestring = filename[:-5] + '%02i.h5' % k
    print(filestring)
    fieldname = filestring[:3]

    try:
        with h5py.File(filestring, mode="r") as my_file:
            scan_id = filestring[-12:-3]# + filename[-5:-3]
            tod_ind = np.array(my_file['tod'][:])
            n_det_ind, n_sb, n_freq, n_samp = tod_ind.shape
            mask_ind = my_file['freqmask'][:]
            mask_full_ind = my_file['freqmask_full'][:]
            pixels = np.array(my_file['pixels'][:]) - 1 
            pix2ind = my_file['pix2ind'][:]
            windspeed = np.mean(my_file['hk_windspeed'][()])
            feature = my_file["feature"][()]
            if feature == 32:
                scanmode = "CES"
            else:
                scanmode = "Lissajous"
            try:
                acc_ind = np.array(my_file['acceptrate'])
            except KeyError:
                acc_ind = np.zeros_like(tod_ind[:,:,0,0])
                print("Found no acceptrate")
            time = np.array(my_file['time'])
            try:
                pca = np.array(my_file['pca_comp'])
                eigv = np.array(my_file['pca_eigv'])
                ampl_ind = np.array(my_file['pca_ampl'])
            except KeyError:
                pca = np.zeros((10, 10000))
                eigv = np.zeros(0)
                ampl_ind = np.zeros((10, n_det_ind, n_sb, 1024))
                print('Found no pca comps')
            try:
                tsys_ind = np.array(my_file['Tsys_lowres'])
            except KeyError:
                tsys_ind = np.zeros_like(tod_ind[:,:,:]) + 40
                print("Found no tsys")
    except:
        print('error loading scanid', k)
        continue
    print(scan_id)

    time = (time - time[0]) * (24 * 60)  # minutes


    n_freq_hr = len(mask_full_ind[0,0])
    n_det = np.max(pixels) + 1 
    # print(n_det)

    ## transform to full arrays with all pixels
    tod = np.zeros((n_det, n_sb, n_freq, n_samp))
    mask = np.zeros((n_det, n_sb, n_freq))
    mask_full = np.zeros((n_det, n_sb, n_freq_hr))
    acc = np.zeros((n_det, n_sb))
    ampl = np.zeros((10, n_det, n_sb, n_freq_hr))
    tsys = np.zeros((n_det, n_sb, n_freq))

    # print(ampl_ind.shape)
    # print(ampl[:, pixels, :, :].shape)

    tod[pixels] = tod_ind
    mask[pixels] = mask_ind
    mask_full[pixels] = mask_full_ind
    acc[pixels] = acc_ind
    ampl[:, pixels, :, :] = ampl_ind
    tsys[pixels] = tsys_ind

    #print(ampl)
    #print(ampl.shape)


    ampl2 = ampl[:] / mask_full[None, :]
    # a = ampl[0, 0, :, :]
    ampl[:, :, (0, 2)] = ampl[:, :, (0, 2), ::-1]
    ampl2[:, :, (0, 2)] = ampl2[:, :, (0, 2), ::-1]
    # ax = fig.add_subplot(611)
    # radiometer *= 40
    #maxval = np.max(np.abs(ampl[0].flatten()))#np.nanstd(np.abs(ampl[0].flatten())) * 2
    for j in range(len(ampl[:, 0, 0, 0])):
        fig = plt.figure(figsize=(12, 10))
        #fig, axes = plt.subplots(21, 2, figsize=(10, 15), gridspec_kw={'width_ratios': [3, 2]})
        gs = fig.add_gridspec(22, 2, width_ratios=[3, 2])
        fig.suptitle('PCA, %s %s scan: ' % (fieldname, scanmode) + scan_id + ', mode:' + str(j+1) + ', eigv.: %.2f' % (eigv[j]) 
                     + ', windspeed: %.2f m/s' % (windspeed), fontsize=12)
        fig.subplots_adjust(top=0.915)
        maxval = 0.15 #max(np.nanstd(np.abs(ampl[j][ampl[j] != 0].flatten())) * 15, 0.01)
        ax3 = fig.add_subplot(gs[:2, :])
        ax3.plot(time, pca[j]) #, label=str(scan_id) + ", PCA comp. " + str(j+1))   # , label='PCA common mode')
        #ax.legend()
        ax3.set_ylabel('pca mode')
        ax3.set_xlim(0, time[-1])
        ax3.set_xlabel('time [m]')
        ax3.set_yticklabels(())
        ax3.set_yticks(())
        ax3.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top') 
        for i in range(19):
            # ax = axes[i+1][0]
            # ax2 = axes[i+1][1]
            ax = fig.add_subplot(gs[i+2, 0])
            ax2 = fig.add_subplot(gs[i+2, 1])
            #ax = fig.add_subplot(20, 2, 1 + 2 * i)
            nu = np.linspace(26, 34, 4096 + 1)[:-1]
            dnu = nu[1] - nu[0]
            nu = nu + dnu / 2
            a = ampl[j, i, :, :].flatten()
            am = ampl2[j, i, :, :].flatten()
            ax.plot(nu, am)
            ax.yaxis.set_label_position("right")
            h = ax.set_ylabel(str(i+1))
            
            ax.set_ylim(-maxval, maxval)
            ax.set_xlim(26, 34)
            ax.set_xlabel('Frequency [GHz]')
            #h.set_rotation(0)

            

            if i < 19:
                ax.set_xticklabels(())
                ax2.set_xticklabels(())
            # if i == 0:
            #     ax.set_title('PCA ampl., scan: ' + scan_id + ', mode:' + str(j+1) + ', eigv.:' + str(eigv[j]) + ', windspeed: ' + str(windspeed), fontsize=12)
            
            f = np.abs(fft.rfftfreq(len(a), dnu * 1e3))[1:]
            #print(f)
            #print(a)
            ps = (np.abs(fft.rfft(a)) ** 2 / len(a))[1:]
            #print(ps)
            ax2.plot(f, ps)

            ax2.set_xscale('log')
            ax2.set_yscale('log')

            ax2.set_xlim(2e-3, 2e-2)
            ax2.set_ylim(1e-7, 1e0)
            if i == 19:
                ax2.set_xlabel('[1/MHz]')
            if i < 19:
                ax2.set_xticks(())

            ax2.xaxis.set_minor_formatter(mticker.NullFormatter())

        #save_string = 'pca_%02i_%s_%07i_%02i_00_AB.png' % (j + 1, date, int(scan_id[:-2]), int(scan_id[-2:]))
        ############## plt.savefig(save_folder + 'acc_chi2/' + save_string, bbox_inches='tight')
        # plt.savefig(save_folder + save_string, bbox_inches='tight')
        plt.savefig(folder + 'pca_ampl_' + scan_id + '_'+ str(j+1) + '.png', bbox_inches='tight')
    # for j in range(len(ampl[:, 0, 0, 0])):
    #     #fig = plt.figure(figsize=(12, 10))
    #     fig, axes = plt.subplots(20, 2, figsize=(10, 15), gridspec_kw={'width_ratios': [3, 2]})
    #     fig.suptitle('PCA ampl., scan: ' + scan_id + ', mode:' + str(j+1) + ', eigv.:' + str(eigv[j]) + ', windspeed: ' + str(windspeed), fontsize=12)
    #     fig.subplots_adjust(top=0.95)
    #     maxval = 0.15 #max(np.nanstd(np.abs(ampl[j][ampl[j] != 0].flatten())) * 15, 0.01)
    #     for i in range(20):
    #         ax = axes[i][0]
    #         ax2 = axes[i][1]
    #         #ax = fig.add_subplot(20, 2, 1 + 2 * i)
    #         nu = np.linspace(26, 34, 4096 + 1)[:-1]
    #         dnu = nu[1] - nu[0]
    #         nu = nu + dnu / 2
    #         a = ampl[j, i, :, :].flatten()
    #         am = ampl2[j, i, :, :].flatten()
    #         ax.plot(nu, am)
    #         ax.yaxis.set_label_position("right")
    #         h = ax.set_ylabel(str(i+1))
            
    #         ax.set_ylim(-maxval, maxval)
    #         ax.set_xlim(26, 34)
    #         ax.set_xlabel('Frequency [GHz]')
    #         #h.set_rotation(0)

            

    #         if i < 19:
    #             ax.set_xticklabels(())
    #             ax2.set_xticklabels(())
    #         # if i == 0:
    #             # ax.set_title('PCA ampl., scan: ' + scan_id + ', mode:' + str(j+1) + ', eigv.:' + str(eigv[j]) + ', windspeed: ' + str(windspeed), fontsize=12)
            
    #         #ax2 = fig.add_subplot(20, 2, 2 + 2 * i)
    #         f = np.abs(fft.rfftfreq(len(a), dnu * 1e3))[1:]
    #         #print(f)
    #         #print(a)
    #         ps = (np.abs(fft.rfft(a)) ** 2 / len(a))[1:]
    #         #print(ps)
    #         ax2.plot(f, ps)
    #         #ax2.loglog(nu, am)
    #         ax2.set_xscale('log')
    #         ax2.set_yscale('log')
    #         # plt.draw()
    #         #ax2.xaxis.set_ticklabels((), color='p')
    #         #plt.setp(ax2.get_xticklabels(), visible=False)
    #         #ax2.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    #         ax2.set_xlim(2e-3, 2e-2)
    #         ax2.set_ylim(1e-7, 1e0)
    #         if i == 19:
    #             ax2.set_xlabel('[1/MHz]')
    #         if i < 19:
    #             ax2.set_xticks(())
    #     #  ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #         #ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    #         ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
    #         #ax2.set_xticks(())
    #         #ax2.set_xticklabels(ax2.get_xticks())
    #         #ax2.xaxis.set_ticks_position('none') 
    #         # labels = ax2.get_xticklabels() 
    #         # for label in labels:
    #         #     print(labels)
    #         # ax2.set_xticklabels(())
    #         # labels = ax2.get_xticklabels() 
    #         # for label in labels:
    #         #     print(labels)
            
            
    #     #plt.tight_layout()
    #     plt.savefig(folder + 'pca_ampl_' + scan_id + '_'+ str(j+1) + '.png', bbox_inches='tight')
