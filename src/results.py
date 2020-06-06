from functions import *
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV,SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
simplefilter(action='ignore')


def ml_alg(X,y):
    print('\nMETA-LEARNER COMPARISON')
    lr_acc = lr(X,y)
    pac_acc = pac(X,y)
    rc_acc = rc(X,y)
    lda_acc = lda(X,y)
    svc_acc = svc(X,y)
    kn_acc = kn(X,y)
    gp_acc = gp(X,y)
    nb_acc = nb(X,y)
    dt_acc = dt(X,y)
    rf_acc = rf(X,y)
    et_acc = et(X,y)
    gbc_acc = gbc(X,y)
    mapping = {lr_acc:'lr_acc',pac_acc:'pac_acc',rc_acc:'rc_acc',
               lda_acc:'lda_acc',svc_acc:'svc_acc',kn_acc:'kn_acc',
               gp_acc:'gp_acc',nb_acc:'nb_acc',dt_acc:'dt_acc',rf_acc:'rf_acc',et_acc:'et_acc',gbc_acc:'gbc_acc'}  
    results = [lr_acc,pac_acc,rc_acc,lda_acc,svc_acc,kn_acc,gp_acc,nb_acc,dt_acc,rf_acc,et_acc,gbc_acc]
    x_plot = ['LR','PAC','RC','LDA','SVC','KN','GP','NB','DT','RF','ET','GBC']
    y_plot = results   
    data_plot = pd.DataFrame({'meta_learner':x_plot,'accuracy':y_plot})
    plt.figure()
    g = sns.barplot(x='meta_learner',y='accuracy',data=data_plot)
    g.set(ylim=(0, 0.70))
    for p in g.patches:
        g.annotate(format(p.get_height(),'.2f'),(p.get_x() + p.get_width()/2.,p.get_height()),ha = 'center',va = 'center',xytext = (0, 10),textcoords = 'offset points')
    plt.show()
    results = np.array(results)
    print('\nMAX/MIN/MEAN OF THE META-LEARNER COMPARISON')
    print('\nMax: ',results.max(),mapping.get(max(mapping)))  
    print('Mean: ',results.mean())
    print('Min: ',results.min())
    print('\n\n')
    
def rfecv_fc(X,y,estimator):
    print('RFECV FEATURE SELECTION:')
    estimator = estimator
    selector = RFECV(estimator, step=1, cv=5)
    og_X = pd.DataFrame(X)
    X = selector.fit_transform(og_X, y)   
    print('Optimal number of features :', selector.n_features_)
    print('Best features index :', og_X.columns[selector.support_])
    print('Best features:')
    for x in og_X.columns[selector.support_]:
        print(tmp[x])
    ml_alg(X,y)
    return og_X.columns[selector.support_].tolist() 


def lasso_fc(X,y):
    print('LASSO FEATURE SELECTION:')
    X = pd.DataFrame(X)
    sel = SelectFromModel(LogisticRegression(C=1, penalty='l1')).fit(X,y)
    sel_feat = X.columns[(sel.get_support())]
    print('Optimal number of features :', len(sel_feat))
    print('Best features index :', sel_feat)
    print('Best features:')
    for i in sel_feat:
        print(tmp[i])
    X = X[sel_feat]   
    X = X.to_numpy()
    ml_alg(X,y) 
    return sel_feat

X = np.load('openml_X.npy')
y = np.load('openml_y.npy')
X = MinMaxScaler().fit_transform(X)


c = Counter(y)
data_perc = [(i, c[i]) for i in c]
print('\n Class Balance',data_perc)


df = pd.DataFrame(X)
df2 = pd.DataFrame(y)

tmp =      ['n_samples','n_features','n_classes','class_weights_min','class_weights_mean','class_weights_max',
           'mean_min','mean_mean','mean_max','t_mean_min','t_mean_mean','t_mean_max','median_min','median_mean','median_max',
           'sem_min','sem_mean','sem_max','std_min','std_mean','std_max','mad_min','mad_mean','mad_max','var_min','var_mean','var_max',
           'skew_min','skew_mean','skew_max','kurt_min','kurt_mean','kurt_max','p_corr_min','p_corr_mean','p_corr_max',
           'k_corr_min','k_corr_mean','k_corr_max','s_corr_min','s_corr_mean','s_corr_max','cov_min','cov_mean','cov_max',
           'variation_min','variation_mean','variation_max','z_score_min','z_score_mean','z_score_max','iqr_min','iqr_mean','iqr_max',
           'iqr_mul_outliers_sum','iqr_mul_outliers_per','iqr_uni_outliers_sum','iqr_uni_outliers_per','z_mul_outliers_sum','z_mul_outliers_per','z_uni_outliers_sum','z_uni_outliers_per',
           'X_entr_min','X_entr_mean','X_entr_max','y_entr','mutual_info_min','mutual_info_mean','mutual_info_max','en','ns',
           'if_an_sum','if_an_per','lof_an_sum','lof_an_per','svm_an_sum','svm_an_per',
           'pca_ev_sum','pca_ev_min','pca_ev_mean','pca_ev_max','pca_sv_sum','pca_sv_min','pca_sv_mean','pca_sv_max','pca_nv',
           'tsvd_ev_sum','tsvd_ev_min','tsvd_ev_mean','tsvd_ev_max','tsvd_sv_sum','tsvd_sv_min','tsvd_sv_mean','tsvd_sv_max','anova_f_min','anova_f_mean','anova_f_max','anova_sum','anova_per',
           'singv_min','singv_mean','singv_max',
           'chi2_based_scores_min','chi2_based_scores_mean','chi2_based_scores_max','rfecv_per_optimal_feat',
           'dt_best_acc','dt_rnd_acc','kn1_acc','lda_acc','nb_acc']
df = pd.DataFrame(X, columns = tmp)
df[df.shape[1]] = df2
corr = df.corr()


ml_alg(X,y)

rfecv_features = rfecv_fc(X,y,SVC(kernel="linear"))
lasso_features = lasso_fc(X,y)

np.random.seed(1)
X_rand = np.random.random((86, 1))
ml_alg(X_rand,y)

print('\nNUMBER OF FEATURES SELECTED BY BOTH RFECV/LASSO:')
print(len(list(set(rfecv_features) & set(lasso_features))))
print('\nFEATURES SELECTED BY BOTH RFECV/LASSO:\n')
for i in list(set(rfecv_features) & set(lasso_features)):
    print(tmp[i])

