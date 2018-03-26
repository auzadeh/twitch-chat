def plot_user_activity(dir):
    df_merge = pd.read_hdf(dir+str(theta)+'_alluser_count_delta.h5','user_count_meanless_mean_greater_delta')
    f,ax= plt.subplots(1,1)
    f.subplots_adjust(hspace=0.001, wspace=0.001)
    idx = (df_merge['delta_norm'] < 0)
    logd = np.log10(df_merge['delta_norm'].abs())
    logd[idx] = -logd[idx]
    logd = logd[~np.isinf(logd)]
    values , bins, _ = plt.hist(df_merge['delta_norm'].tolist(),bins=20, normed=True,alpha=0.5,color='k')
    plt.xlabel('Delta',fontsize='x-large')
    plt.ylabel('Density',fontsize='x-large')
    plt.savefig(str(theta)+'_alluser_delta_hist.png')
    plt.show()
    values , bins = np.histogram(df_merge['delta_norm'].tolist(),bins=20,  normed=True)
    bins_diff = np.diff(bins)
    bin_width = bins_diff[0]
    area_accum = []
    bin_index = []
    indx = 0
    for i in range(0,10):
        area_accum.append(sum(bin_width*values[10-i:10+i]))
        bin_index.append(i)
    plt.plot(bin_index,area_accum,ls='',marker='o',c='k',markerfacecolor="None")
    plt.xlabel('$\epsilon$',fontsize='x-large')
    plt.ylabel('Area',fontsize='x-large')
    plt.savefig(str(theta)+'_area_alluser_delta.png')
    plt.show()

def plot_devide(theta,dir):
    df = pd.read_table(dir+'sample_2_user_slopes_norm_all_4.csv',names=['user','slope_l','slope_g'],sep=',')
    df.dropna(inplace =True)
    nullfmt = NullFormatter()
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    x = df['slope_l'].astype(float).tolist()
    y = df['slope_g'].astype(float).tolist()
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    plt.figure(1, figsize=(8, 8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = 10.0 * (Z2 - Z1)
    CS = axScatter.contour(X, Y, Z)
    axScatter.clabel(CS, inline=1, fontsize=10)
    axScatter.axhline(linewidth=2, color='k')
    axScatter.axvline(linewidth=2, color='k')
    binwidth = 0.25
    axHistx.hist(x, bins=50,facecolor='Navy',normed = 1,alpha=0.63)#bins)
    axHisty.hist(y, bins=50, orientation='horizontal',facecolor = 'Navy',alpha=0.63,normed = 1)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    ln_p_n = df[df['slope_l']>0]
    ln_p_n = ln_p_n[ln_p_n['slope_g']<0]
    axScatter.text(0.5,-0.5,str(round(float(len(ln_p_n))/len(df),2)),color='r',fontsize='x-large')
    ln_p_p =  df[df['slope_l']>0]
    ln_p_p = ln_p_p[ln_p_p['slope_g']>0]
    axScatter.text(0.5,0.5,str(round(float(len(ln_p_p))/len(df),2)),color='r',fontsize='x-large')
    ln_n_n =  df[df['slope_l']<0]
    ln_n_n = ln_n_n[ln_n_n['slope_g']<0]
    axScatter.text(-0.5,-0.5,str(round(float(len(ln_n_n))/len(df),2)),color='r',fontsize='x-large')
    ln_n_p =  df[df['slope_l']<0]
    ln_n_p = ln_n_p[ln_n_p['slope_g']>0]
    axScatter.text(-0.5,0.5,str(round(float(len(ln_n_p))/len(df),2)),color='r',fontsize='x-large')
    plt.show()



def sample_balanced_user(dir,theta):
    sample_size = 1000
    df_merge = pd.read_hdf(dir+str(theta)+'_alluser_count_delta.h5','user_count_meanless_mean_greater_delta')
    df_merge = df_merge[df_merge['delta'] == 0.0]
    df_merge['sum'] = df_merge['sum_less']+df_merge['sum_greater']
    df_merge = df_merge[df_merge['sum'] >100]
    df_merge = df_merge.sort(['sum'],ascending=[0])
    users_list = df_merge['username'].tolist()
    user_sample = []
    l = len(users_list)/4
    for i in range(4):
        user_sample += np.random.choice(users_list[i*l: (i+1)*l],sample_size/4).tolist()
    df = pd.read_hdf(dir+'msgcount_username_usercount_userprob.h5','msgcount_username_usercount_userprob')
    df = df[df['username'].isin(user_sample)]
    df_group = pd.DataFrame({'sum' : df.groupby(['username'])['msgcount'].sum()}).reset_index()
    sample_users = []
    x_y_dic ={}
    for i in range(21):
        x_y_dic[i] = []
    for u in user_sample:
        df_u = df[df['username']==u]
        df_u = df_u.sort(['msgcount'])
        sample_users.append(df_u)
        x,y,std = log_binning(df_u['msgcount'].tolist(),df_u['usercount'].tolist(),bin_count=20)
        for i in range(len(x)):
            x_y_dic[x[i]].append(y[i])
        plt.plot(x,y,c='k',alpha=0.1)
        plt.yscale('log')
    x_list = []
    y_list = []
    y_err = []
    y_delta = []
    x_delta = []
    for i in x_y_dic:
        x_list.append(i)
        y_list.append(np.mean(x_y_dic[i]))
        y_err.append(stats.sem(x_y_dic[i]))
        for j in x_y_dic[i]:
            y_delta.append(j-np.mean(x_y_dic[i]))
            x_delta.append(i)
    plt.errorbar(x_list,y_list,y_err,c='k',ls='',fmt='o',mfc='r',markersize=10,barsabove=0,capsize=0,elinewidth=0)
    plt.savefig('df_sample_unw_users_deltae0p0.png')
    plt.show()
    plt.scatter(x_delta,y_delta,alpha=0.2)
    plt.savefig('df_sample_delta_logavr_e0p0.png')
    plt.show()
    plt.hist(np.sqrt(y_delta),bins=20, normed=True,alpha=0.5,color='k')
    plt.savefig('hist_delta_avr_e0p0.png')
    plt.show()

def plot_users(dir):
    df = pd.read_hdf(dir+str(theta)+'_msgcount_username_usercount_userprob.h5','msgcount_username_usercount_userprob')
    df_group = pd.DataFrame({'sum' : df.groupby(['username'])['msgcount'].sum()}).reset_index()
    df_group = df_group.sort(['sum'],ascending=[0])
    users = df_group['username'].tolist()[1:10]
    for u in users:
        df_u = df[df['username']==u]
        df_u = df_u.sort(['msgcount'])
        plt.plot(df_u['msgcount'].tolist(),df_u['usercount'].tolist(),ls='',marker='o',c='k',alpha=0.2)
        plt.xlabel('msg count',fontsize='x-large')
        plt.ylabel('usercount',fontsize='x-large')
        plt.savefig(str(theta)+'_'+str(u)+'_intersect.png')
        plt.clf()

def plot_users_notintersect(dir):
    df = pd.read_hdf(dir+'not_msgcount_username_usercount_userprob.h5','not_msgcount_username_usercount_userprob')
    df_group = pd.DataFrame({'sum' : df.groupby(['username'])['msgcount'].sum()}).reset_index()
    df_group = df_group.sort(['sum'],ascending=[0])
    users = df_group['username'].tolist()[1:10]
    for u in users:
        df_u = df[df['username']==u]
        df_u = df_u.sort(['msgcount'])
        plt.plot(df_u['msgcount'].tolist(),df_u['usercount'].tolist(),ls='',marker='o',c='k',alpha=0.2)
        plt.xlabel('msg count',fontsize='x-large')
        plt.ylabel('usercount',fontsize='x-large')
        plt.savefig(str(u)+'_notintersect.png')
        plt.clf()

def plot_users_of_overload(dir):
    df = pd.read_hdf(dir+'msgcount_username_usercount_userprob.h5','msgcount_username_usercount_userprob')
    df = df.sort(['msgcount'],ascending=[0])
    one_u = df[df['msgcount']==11579.0]
    df_one_u = one_u.sort(['usercount'],ascending=[0])
    users = df_one_u['username'].tolist()[1:10]
    for u in users:
        df_u = df[df['username']==u]
        df_u = df_u.sort(['msgcount'])
        plt.plot(df_u['msgcount'].tolist(),df_u['usercount'].tolist(),ls='',marker='o',c='k',alpha=0.2)
        plt.xlabel('msg count',fontsize='x-large')
        plt.ylabel('usercount',fontsize='x-large')
        plt.savefig(str(u)+'_msg_11579.png')

