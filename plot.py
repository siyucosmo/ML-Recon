import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from nbodykit.lab import *
from pmesh.pm import ParticleMesh
from matplotlib.gridspec import GridSpec

dk = 0.02

c = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss(path):
    loss_train = np.loadtxt(path+'/trainLoss.txt')
    loss_val = np.loadtxt(path+'/valLoss.txt')
    plt.figure()
    plt.plot(loss_train,label='train')
    plt.plot(np.arange(1,len(loss_val)+1)*20,loss_val,label='val')
    plt.legend()
    plt.savefig(path+'/loss.png')


def get_parser():
    parser = argparse.ArgumentParser(description="config for plot")
    parser.add_argument('--config_file_path',type=str,default='')
    return parser

#---------get powspec-------------#
def getPow(data1,data2=None):
    pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
    q = pm.generate_uniform_particle_grid()
    den1 = pm.paint(q+data1.reshape([-1,3]))
    if(data2 is not None):
        pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
        den2 = pm.paint(q+data2.reshape([-1,3]))
        temp = FFTPower(first=den1, second=den2,mode='1d',BoxSize=128,dk=dk)
        k,powspec = temp.power['k'],temp.power['power']
    else:
        temp = FFTPower(den1, mode='1d',BoxSize=128,dk=dk)
        k,powspec = temp.power['k'],temp.power['power']
    return [k,powspec.real]

def getPow_dis(data1,data2=None):
    pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
    q = pm.generate_uniform_particle_grid()
    den1 = pm.paint(q)
    power = 0
    if(data2 is not None):
        pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
        q = pm.generate_uniform_particle_grid()
        den2 = pm.paint(q)
        for ii in range(3):
            den1[:] = data1[:,:,:,ii]
            den2[:] = data2[:,:,:,ii]
            temp = FFTPower(first = den1, second = den2, mode='1d',BoxSize=128,dk=dk)
            k,power = temp.power['k'], power+temp.power['power']
    else:
        for ii in range(3):
            den1[:] = data1[:,:,:,ii]
            temp = FFTPower(den1, mode='1d',BoxSize=128,dk=dk)
            k,power = temp.power['k'], power+temp.power['power']
    return [k,power.real]

def getPow_ave(path_pred,path_true,f_pred_pre,f_pred_post,f_true,num,d):
    powLPT,powNbody,powRecon,powReconxNbody,powLPTxNbody = 0,0,0,0,0
    true = np.fromfile(path_true+f_true,dtype='f4').reshape(-1,32,32,32,10)
    LPT = true[:,:,:,:,4:7]
    Nbody = true[:,:,:,:,7::]
    if(d==1):
        for n in range(0,num):
            test = np.einsum('ijkl->jkli',np.load(path_pred+f_pred_pre+str(n)+f_pred_post))[:,:,:,0:3]
            k,P = getPow_dis(LPT[n])
            powLPT += P
            k,P = getPow_dis(Nbody[n])
            powNbody += P
            k,P = getPow_dis(test)
            powRecon += P
            k,P = getPow_dis(test,Nbody[n])
            powReconxNbody += P
            k,P = getPow_dis(LPT[n],Nbody[n])
            powLPTxNbody += P
    else:
        for n in range(0,num):
            test = np.einsum('ijkl->jkli',np.load(path_pred+f_pred_pre+str(n)+f_pred_post))[:,:,:,0:3]
            k,P = getPow(LPT[n])
            powLPT += P
            k,P = getPow(Nbody[n])
            powNbody += P
            k,P = getPow(test)
            powRecon += P
            k,P = getPow(test,Nbody[n])
            powReconxNbody += P
            k,P = getPow(LPT[n],Nbody[n])
            powLPTxNbody += P
    return k,powNbody/num,powLPT/num,powRecon/num,powLPTxNbody/num,powReconxNbody/num

#----------plot powspec--------------#
def plot_pow(k,powNbody,powLPT,powRecon,LxN,RxN,title):
    fig = plt.figure(figsize=(6,8))

    ax1 = plt.subplot2grid((4,1),(0,0),rowspan=2)
    plt.plot(k, powLPT,label = '2LPT')
    plt.plot(k, powRecon,label = 'U-Net')
    plt.plot(k, powNbody,label ='fastPM')
    plt.ylabel('P(k)')
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.title(title)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2 = plt.subplot2grid((4,1),(2,0), rowspan = 1,sharex=ax1)
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(k, powLPT/powNbody,label = 'LPT')
    plt.plot(k, powRecon/powNbody,label = 'Predict')
    plt.ylabel(r'$T(k)$')
    plt.setp(ax2.get_xticklabels(),visible=False)

    ax3 = plt.subplot2grid((4,1),(3,0),sharex=ax1)
    plt.loglog(k, 1-(LxN/np.sqrt(powLPT*powNbody))**2,label = 'LPTxNbody')
    plt.loglog(k, 1-(RxN/np.sqrt(powRecon*powNbody))**2,label = 'ReconxNbody')
    plt.xticks(np.round([0.06+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2),
               np.round([0.06+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2))
    plt.xticks(rotation=45)
    plt.ylabel(r'1-$r^2$')
    #plt.tight_layout()

def plot_pancake(k,powNbody,powRecon,powInput,title):
    pos = np.intersect1d(np.where(np.nan_to_num(powNbody) > 1e-3)[0], np.where(np.nan_to_num(powRecon)> 1e-3)[0])
    plt.figure(figsize=(6,6))
    gs = GridSpec(2, 1, height_ratios=[2,1],width_ratios=[1])
    ax1 = plt.subplot(gs[0, 0])
    #plt.loglog(k[pos],powLPT[pos],'*',label='2LPT',c=c[0])
    plt.loglog(k[pos],powRecon[pos],'*',label='U-Net',c=c[1])
    plt.loglog(k[pos],powNbody[pos],'x',label='FastPM',c=c[2])
    plt.loglog(k,powInput,'^',label='1 mode input',c=c[3])
    plt.vlines(k[np.argmax(np.nan_to_num(powInput))],\
               ymin = 1e-5,\
               ymax = powInput[np.argmax(np.nan_to_num(powInput))],alpha=0.5,linestyles='dashed')
    plt.legend(loc='upper left')
    plt.title(r"displacement "+title)
    plt.ylabel(r'$P(k)}$')
    plt.ylim(ymin = 1e-5)

    plt.subplot(gs[1, 0],sharex=ax1)
    plt.loglog(k[pos],powRecon[pos]/powNbody[pos],'*',label='Recon',c=c[1])
    plt.xlabel('k '+r'[h/Mpc]')
    plt.ylim(ymin = 1e-5)
    plt.ylabel(r'$T(k)$')
    plt.axhline(y=1,color='black',ls='dashed')
    plt.setp(ax1.get_xticklabels(),visible=False)
    plt.xticks(np.round([0.06+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2),\
                   np.round([0.06+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2))
    plt.xticks(rotation=45)
    #plt.tight_layout()

#----------plot one slice demonstration--------------#
def look_den_slice(net1,net2,net3,s,title):
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    matplotlib.rc('font',size=12)
    cmap = 'coolwarm'
    assert net1.shape[-1] == 3
    pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
    q = pm.generate_uniform_particle_grid()
    net1 = pm.paint(q+net1.reshape([-1,3]))
    pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
    net2 = pm.paint(q+net2.reshape([-1,3]))
    pm = ParticleMesh(BoxSize=128, Nmesh=[32, 32, 32])
    net3 = pm.paint(q+net3.reshape([-1,3]))
    fig = plt.figure(figsize=(6,4))
    amp_low = min(np.percentile(net1[:,:,s],5),
                  np.percentile(net2[:,:,s],5),
                  np.percentile(net3[:,:,s],5))
    amp_high = max(np.percentile(net1[:,:,s],95),
                  np.percentile(net2[:,:,s],95),
                  np.percentile(net3[:,:,s],95))
    amp_low1 = min(np.percentile((net1[:,:,s]-net2[:,:,s]),5),
                  np.percentile((net1[:,:,s]-net3[:,:,s]),5))
    amp_high1= max(np.percentile((net1[:,:,s]-net2[:,:,s]),95),
                  np.percentile((net1[:,:,s]-net3[:,:,s]),95))
    amp = max(np.abs(amp_low1),amp_high1)
    gs = GridSpec(2, 4, height_ratios=[1,1],width_ratios=[1,1,1,0.1])
    plt.subplot(gs[0, 0])
    im = plt.imshow(net1[:, :, s],cmap =cmap,vmin = amp_low,vmax=amp_high)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title('fastPM')
    plt.subplot(gs[0, 1])
    plt.imshow(net2[:, :, s],cmap = cmap,vmin = amp_low,vmax=amp_high)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title('2LPT')
    plt.subplot(gs[0, 2])
    plt.imshow(net3[:, :, s],cmap = cmap,vmin = amp_low,vmax=amp_high)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title('U-Net')
    cbax = plt.subplot(gs[0,3])
    cbar = fig.colorbar(mappable=im,cax=cbax, orientation = 'vertical',
                 ticklocation = 'right')
    cbar.ax.tick_params(labelsize=12)
    plt.subplot(gs[1, 1])
    im = plt.imshow((net1[:, :, s]-net2[:, :, s]),cmap = cmap,vmin = -amp,vmax=amp)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title(r'fastPM $-$ 2LPT')
    plt.subplot(gs[1, 2])
    plt.imshow((net1[:, :, s]-net3[:, :, s]),cmap = cmap,vmin = -amp,vmax=amp)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title(r'fastPM $-$ U-Net')
    cbax = plt.subplot(gs[1,3])
    cbar = fig.colorbar(mappable=im,cax=cbax, orientation = 'vertical',
                 ticklocation = 'right')
    cbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(hspace=.2)

def look_dis_slice(net1,net2,net3,s,title):
    #matplotlib.rc('xtick', labelsize=12)
    #matplotlib.rc('ytick', labelsize=12)
    plt.axis('off')
    matplotlib.rc('font',size=12)
    assert net1.shape[-1] == 3
    fig = plt.figure(figsize=(6,4))
    amp_low = min(np.percentile(net1[s,:,:,0],5),
                  np.percentile(net2[s,:,:,0],5),
                  np.percentile(net3[s,:,:,0],5))
    amp_high = max(np.percentile(net1[s,:,:,0],95),
                  np.percentile(net2[s,:,:,0],95),
                  np.percentile(net3[s,:,:,0],95))
    amp_low1 = min(np.percentile((net1[s,:,:,0]-net2[s,:,:,0]),5),
                  np.percentile((net1[s,:,:,0]-net3[s,:,:,0]),5))
    amp_high1 = max(np.percentile((net1[s,:,:,0]-net2[s,:,:,0]),95),
                  np.percentile((net1[s,:,:,0]-net3[s,:,:,0]),95))
    amp = min(np.abs(amp_low),amp_high)
    gs = GridSpec(2, 4, height_ratios=[1,1],width_ratios=[1,1,1,0.1])
    plt.subplot(gs[0, 0])
    im = plt.imshow(net1[s,:,:,0],cmap = "coolwarm",vmin = -amp,vmax=amp)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title('fastPM')
    plt.subplot(gs[0, 1])
    plt.imshow(net2[s,:,:,0],cmap = "coolwarm",vmin = -amp,vmax=amp)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title('2LPT')
    plt.subplot(gs[0, 2])
    plt.imshow(net3[s,:,:,0],cmap = "coolwarm",vmin = -amp,vmax=amp)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title('U-Net')
    cbax = plt.subplot(gs[0,3])
    cbar = fig.colorbar(mappable=im,cax=cbax, orientation = 'vertical',
                 ticklocation = 'right')
    cbar.ax.tick_params(labelsize=12)
    plt.subplot(gs[1, 1])
    im = plt.imshow(net1[s,:,:,0]-net2[s,:,:,0],cmap = "coolwarm",vmin = amp_low1,vmax=amp_high1)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title(r'fastPM $-$ 2LPT')
    plt.subplot(gs[1, 2])
    plt.imshow(net1[s,:,:,0]-net3[s,:,:,0],cmap = "coolwarm",vmin = amp_low1,vmax=amp_high1)
    #plt.xticks([0,16,32],[0,64,128])
    #plt.yticks([16,0],[64,128])
    plt.axis('off')
    plt.title(r'fastPM $-$ U-Net')
    cbax = plt.subplot(gs[1,3])
    cbar = fig.colorbar(mappable=im,cax=cbax, orientation = 'vertical',
                 ticklocation = 'right')
    cbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(hspace=0.2)

#------------plot different A/Om-----------------#
def plot_powA(k,powNbody,powLPT,powRecon,LxN,RxN,label,c_i):
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax1.loglog(k,powLPT,color = c[c_i],ls='--')
    ax1.loglog(k,powRecon,color=c[c_i],ls=':')
    l0=ax1.loglog(k,powNbody,color=c[c_i],ls='-',label=label)

    l1 = ax2.plot(k, powLPT/powNbody,color = c[c_i],ls='--')
    l2 = ax2.plot(k, powRecon/powNbody,label = label,color = c[c_i],ls=':')
    ax2.axhline(y=1, color='k', linestyle='--')


    ax3.loglog(k, 1-(LxN/np.sqrt(powLPT*powNbody))**2,color = c[c_i],ls='--')
    ax3.loglog(k, 1-(RxN/np.sqrt(powRecon*powNbody))**2,color = c[c_i],ls=':')
    return l0,l1,l2
#----------plot residual -----------------#
def plot_residual(noise):
	matplotlib.rc('font',size=12)
	noise = np.einsum('ijkl->jkli',noise)
	fig = plt.figure(figsize=(6,2))
	gs = GridSpec(2, 3, height_ratios=[1,0.05],width_ratios=[1,1,1])
	plt.subplot(gs[0, 0])
	im = plt.imshow(noise[:,:,:,0].mean(axis=0),cmap = "coolwarm")
	plt.title('x')
	#plt.xticks([0,16,32],[0,64,128])
	#plt.yticks([16,0],[64,128])
	plt.axis('off')

	ax = plt.subplot(gs[1,0])
	cbar=plt.colorbar(im,cax=ax,orientation="horizontal",
	                  ticks=[np.min(noise[:,:,:,0].mean(axis=0)),np.max(noise[:,:,:,0].mean(axis=0))])
	cbar.ax.set_xticklabels([np.round(np.min(noise[:,:,:,0].mean(axis=0)),3),
	                         np.round(np.max(noise[:,:,:,0].mean(axis=0)),3)],rotation=30)

	plt.subplot(gs[0, 1])
	im = plt.imshow(noise[:,:,:,1].mean(axis=1),cmap = "coolwarm")
	plt.title('y')
	#plt.xticks([0,16,32],[0,64,128])
	#plt.yticks([16,0],[64,128])
	plt.axis('off')

	ax = plt.subplot(gs[1,1])
	cbar=plt.colorbar(im,cax=ax,orientation="horizontal",
	                  ticks=[np.min(noise[:,:,:,1].mean(axis=1)),np.max(noise[:,:,:,1].mean(axis=1))])
	cbar.ax.set_xticklabels([np.round(np.min(noise[:,:,:,1].mean(axis=1)),3),
	                         np.round(np.max(noise[:,:,:,1].mean(axis=1)),3)],rotation=30)

	plt.subplot(gs[0, 2])
	im = plt.imshow(noise[:,:,:,2].mean(axis=2),cmap = "coolwarm")
	plt.title('z')
	#plt.xticks([0,16,32],[0,64,128])
	#plt.yticks([16,0],[64,128])
	plt.axis('off')

	ax = plt.subplot(gs[1,2])
	cbar=plt.colorbar(im,cax=ax,orientation="horizontal",
	                  ticks=[np.min(noise[:,:,:,2].mean(axis=2)),np.max(noise[:,:,:,2].mean(axis=2))])
	cbar.ax.set_xticklabels([np.round(np.min(noise[:,:,:,2].mean(axis=2)),3),
	                         np.round(np.max(noise[:,:,:,2].mean(axis=2)),3)],rotation=30)
	plt.subplots_adjust(wspace=0.2)

	plt.savefig(configs['res']['pred_path']+'noise.pdf',bbox_inches='tight')

if __name__ == "__main__":
    print (matplotlib.matplotlib_fname())
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_file_path) as f:
        configs = json.load(f)

    if configs["loss"]:
        plot_loss(configs["path"])
    #plot pow
    elif configs['pow']["plot"]:
        k,powNbody,powLPT,powRecon,powLPTxNbody,powReconxNbody= getPow_ave(configs["pow"]["path_pred"],configs["pow"]["path_true"],'test_','.npy','00000000-00001000.32.10.f4',1000,1)
        print (k,powRecon/powNbody)
        plot_pow(k,powNbody,powLPT,powRecon,powLPTxNbody,powReconxNbody,'displacement')
        plt.savefig(configs["pow"]["path_pred"]+"PDD.pdf",bbox_inches='tight')
        k,powNbody,powLPT,powRecon,powLPTxNbody,powReconxNbody= getPow_ave(configs["pow"]["path_pred"],configs["pow"]["path_true"],'test_','.npy','00000000-00001000.32.10.f4',1000,0)
        print (k,powRecon/powNbody)
        plot_pow(k,powNbody,powLPT,powRecon,powLPTxNbody,powReconxNbody,'density')
        plt.savefig(configs["pow"]["path_pred"]+"Pmm.pdf",bbox_inches='tight')
    #plot pancake
    elif configs['pancake']['plot']:
        pancake_Nbody = np.fromfile(configs["pancake"]["path_true"]+"00-00-"+configs['pancake']['wave_num']+"-phi090/"+\
                            '00000000-00001000.32.10.f4',dtype='f4').reshape(-1,32,32,32,10)
        k,powInput = getPow_dis(pancake_Nbody[0][:,:,:,1:4])
        k,powNbody,powLPT,powRecon,powLPTxNbody,powReconxNbody= getPow_ave(configs['pancake']['path_pred'],configs['pancake']['path_true']+"00-00-"+configs['pancake']['wave_num']+"-phi090/",'pancake_00-00-'+configs['pancake']['wave_num']+'-test_','.npy','00000000-00001000.32.10.f4',1000,1)
        plot_pancake(k,powNbody,powRecon,powInput)
        plt.savefig(configs['pancake']['path_pred']+"PDD_k_"+configs['pancake']['wave_num']+"_cut.pdf")
    #plot a slice
    elif configs['slice']['plot']:
        test = np.einsum('ijkl->jkli',np.load(configs["slice"]["path_pred"]+'test_'+str(configs["slice"]["n"])+'.npy'))
        true = np.fromfile(configs["slice"]["path_true"]+"00000000-00001000.32.10.f4",dtype='f4').reshape(-1,32,32,32,10)[configs["slice"]["n"],:,:,:,4:7]
        look_dis_slice(test[:, :, :, 3:6],true,test[:, :, :, 0:3],20,"displacement slice")
        plt.savefig(configs["slice"]["path_pred"]+"displacement_slice.pdf")
        look_den_slice(test[:, :, :, 3:6],true,test[:, :, :, 0:3],20,"density slice")
        plt.savefig(configs["slice"]["path_pred"]+"density_slice.pdf")
    #plot different A
    elif configs['A']['plot']:
        A_list = [1.8,1.2,0.8,0.2]
        plt.figure(figsize=(6,8))
        ax1 = plt.subplot2grid((4,1),(0,0),rowspan=2)
        ax2 = plt.subplot2grid((4,1),(2,0),rowspan=1,sharex=ax1)
        ax3 = plt.subplot2grid((4,1),(3,0),rowspan=1,sharex=ax1)
        c_i = 0
        for A in A_list:
            print (A)
            if(configs['A']['d']==1):
                k,powNbody,powLPT,powRecon,LxN,RxN= getPow_ave(configs["A"]["path_pred"],configs["A"]["path_true"]+'0'+str(A)+'0/','0'+str(A)+'0_test_','.npy','00000000-00001000.32.10.f4',1000,1)
            else:
                k,powNbody,powLPT,powRecon,LxN,RxN= getPow_ave(configs["A"]["path_pred"],configs["A"]["path_true"]+'0'+str(A)+'0/','0'+str(A)+'0_test_','.npy','00000000-00001000.32.10.f4',1000,0)
            print (k,powLPT/powNbody,powRecon/powNbody)
            l0,l1,l2 = plot_powA(k,powNbody,powLPT,powRecon,LxN,RxN,str(A)+"A",c_i)
            c_i += 1
        if configs["A"]["d"]==1:
            ax1.set_title('displacement')
        else:
            ax1.set_title('density')
        legend1 = ax1.legend([l0[0],l1[0],l2[0]], ["fastPM","2LPT", "U-Net"],loc=3)
        ax1.legend(loc='upper right')
        ax1.add_artist(legend1)
        ax1.set_ylabel(r'$P(k)$')
        ax2.set_ylabel(r'$T(k)$')
        ax3.set_ylabel(r'$1-r^2$')
        ax3.set_xlabel('k [h/Mpc]')
        plt.setp(ax1.get_xticklabels(),visible=False)
        plt.setp(ax2.get_xticklabels(),visible=False)
        plt.xticks(np.round([0.05+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2),
                 np.round([0.05+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        if configs["A"]["d"]==1:
            plt.savefig(configs["A"]["path_pred"]+"pow_DD_A_predict.pdf",bbox_inches='tight')
        else:
            plt.savefig(configs["A"]["path_pred"]+"pow_mm_A_predict.pdf",bbox_inches='tight')
    #plot different Om
    elif configs['Om']['plot']:
        list_Om = [0.28,0.30,0.32]
        plt.figure(figsize=(6,8))
        ax1 = plt.subplot2grid((4,1),(0,0),rowspan=2)
        ax2 = plt.subplot2grid((4,1),(2,0),rowspan=1,sharex=ax1)
        ax3 = plt.subplot2grid((4,1),(3,0),rowspan=1,sharex=ax1)
        c_i = 0
        for Om in list_Om:
            print (Om)
            if(configs['Om']['d']==1):
                k,powNbody,powLPT,powRecon,LxN,RxN= getPow_ave(configs["Om"]["path_pred"],configs["Om"]["path_true"],'Om_'+str(Om).ljust(4,'0')+'_test_','.npy','om-'+str(Om).ljust(4,'0')+'-'+'00000000-00001000.32.10.f4',1000,1)
            else:
                k,powNbody,powLPT,powRecon,LxN,RxN= getPow_ave(configs["Om"]["path_pred"],configs["Om"]["path_true"],'Om_'+str(Om).ljust(4,'0')+'_test_','.npy','om-'+str(Om).ljust(4,'0')+'-'+'00000000-00001000.32.10.f4',1000,0)
            l0,l1,l2 = plot_powA(k,powNbody,powLPT,powRecon,LxN,RxN,r"$\Omega_m$="+str(Om),c_i)
            c_i += 1

        if configs["Om"]["d"]==1:
            ax1.set_title('displacement')
        else:
            ax1.set_title('density')
        legend1 = ax1.legend([l0[0],l1[0],l2[0]], ["fastPM","2LPT", "U-Net"],loc=3)
        ax1.legend(loc='upper right')
        ax1.add_artist(legend1)
        ax1.set_ylabel(r'$P(k)$')
        ax2.set_ylabel(r'$T(k)$')
        ax3.set_ylabel(r'$1-r^2$')
        ax3.set_xlabel('k [h/Mpc]')
        plt.setp(ax1.get_xticklabels(),visible=False)
        plt.setp(ax2.get_xticklabels(),visible=False)
        plt.xticks(np.round([0.05+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2),
                 np.round([0.05+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        if configs["Om"]["d"]==1:
            plt.savefig(configs["Om"]["path_pred"]+"pow_DD_Om_predict.pdf",bbox_inches='tight')
        else:
            plt.savefig(configs["Om"]["path_pred"]+"pow_mm_Om_predict.pdf",bbox_inches='tight')
    #plot different smoothing
    elif configs['sm']['plot']:
        plt.figure(figsize=(6,8))
        list_sm = [4,8]
        ax1 = plt.subplot2grid((4,1),(0,0),rowspan=2)
        ax2 = plt.subplot2grid((4,1),(2,0),rowspan=1,sharex=ax1)
        ax3 = plt.subplot2grid((4,1),(3,0),rowspan=1,sharex=ax1)
        c_i = 0
        for sm in list_sm:
            print (sm)
            if(configs['sm']['d']==1):
                k,powNbody,powLPT,powRecon,LxN,RxN= getPow_ave(configs["sm"]["path_pred"],configs["sm"]["path_true"],'sm_'+str(sm)+'_test_','.npy','sm-'+str(sm).rjust(2,'0')+'.00/'+'00000000-00001000.32.10.f4',1000,1)
            else:
                k,powNbody,powLPT,powRecon,LxN,RxN= getPow_ave(configs["sm"]["path_pred"],configs["sm"]["path_true"],'sm_'+str(sm)+'_test_','.npy','sm-'+str(sm).rjust(2,'0')+'.00/'+'00000000-00001000.32.10.f4',1000,0)
            l0,l1,l2 = plot_powA(k,powNbody,powLPT,powRecon,LxN,RxN,r"s="+str(sm),c_i)
            c_i += 1

        if configs["sm"]["d"]==1:
            ax1.set_title('displacement')
        else:
            ax1.set_title('density')
        legend1 = ax1.legend([l0[0],l1[0],l2[0]], ["fastPM","2LPT", "U-Net"],loc=3)
        ax1.legend(loc='upper right')
        ax1.add_artist(legend1)
        ax1.set_ylabel(r'$P(k)$')
        ax2.set_ylabel(r'$T(k)$')
        ax3.set_ylabel(r'$1-r^2$')
        ax3.set_xlabel('k [h/Mpc]')
        plt.setp(ax1.get_xticklabels(),visible=False)
        plt.setp(ax2.get_xticklabels(),visible=False)
        plt.xticks(np.round([0.05+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2),
                 np.round([0.05+0.01*i for i in range(0,4,2)]+ [0.1+0.1*i for i in range(0,7,2)],2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        if configs["sm"]["d"]==1:
            plt.savefig(configs["sm"]["path_pred"]+"pow_DD_sm_predict.pdf",bbox_inches='tight')
        else:
            plt.savefig(configs["sm"]["path_pred"]+"pow_mm_sm_predict.pdf",bbox_inches='tight')
    elif configs['dual']['plot']:
        pancake_Nbody = np.fromfile(configs["dual"]["path_true"]+configs['dual']['wave_num1']+"-phi090-"+configs['dual']['wave_num2']+"-phi090/"+\
                            '00000000-00001000.32.10.f4',dtype='f4').reshape(-1,32,32,32,10)
        k,powInput = getPow_dis(pancake_Nbody[0][:,:,:,1:4])
        k,powNbody,powLPT,powRecon,powLPTxNbody,powReconxNbody= getPow_ave(configs['dual']['path_pred'],configs['dual']['path_true']+configs['dual']['wave_num1']+"-phi090-"+configs['dual']['wave_num2']+"-phi090/",'k_'+configs['dual']['wave_num1']+'_k_'+configs['dual']['wave_num2']+'_','.npy','00000000-00001000.32.10.f4',1,1)
        plot_pancake(k,powNbody,powRecon,powInput,"("+r"$k1_x$="+str(np.round(k[np.argmax(np.nan_to_num(powInput))],2))+r" $k2_y$="+str(np.round(np.argsort(np.nan_to_num(np.c_[k,powInput]),axis=-1)[1,0],2))+")")
        plt.savefig(configs['dual']['path_pred']+"PDD_k_"+configs['dual']['wave_num1']+"_k_"+configs['dual']['wave_num2']+".pdf")
    elif configs['var']['plot']:
        plt.figure()
        #x = np.load(configs['var']['path']+'dis_var_'+str(configs["var"]["scale"])+'.npy')
        #plt.hist(x,bins=100,density=True,histtype='step',label = '1A')
        for A in [1.8,1.2,0.8,0.2,1]:
            x = np.load(configs['var']['path']+'dis_var_'+str(A)+'A_'+str(configs["var"]["scale"])+'.npy')
            plt.hist(x,bins=100,density=True,histtype='step',label = str(A)+'A')
        plt.legend()
        plt.xlabel('displacement variance')
        plt.ylabel('density')
        plt.xlim(xmin=0,xmax = x.max()/2)
        plt.title("scale = " +str(configs["var"]["scale"]*4)+" Mpc/h")
        plt.savefig(configs['var']['path']+"hist_var_"+str(configs["var"]["scale"])+".pdf")
    elif configs['res']['plot']:
        noise = np.load(configs['res']['pred_path']+'A_0_k_0_phi_0.npy')
        plot_residual(noise)




