import numpy as np
import pandas as pd  
from scipy import stats
from scipy.stats import entropy
from scipy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor   
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest,RFECV,f_classif,mutual_info_classif,chi2
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.model_selection import cross_val_score
from collections import Counter
from warnings import simplefilter
simplefilter(action='ignore')

cv = 5
def lr(X,y,solver='lbfgs',penalty='l2',C=1,random_state=1):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver=solver,penalty=penalty,C=C,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("LR:", accuracy)
    return accuracy

def pac(X,y,C=1,random_state=1):
    from sklearn.linear_model import PassiveAggressiveClassifier
    model = PassiveAggressiveClassifier(C=C,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("PAC:", accuracy)
    return accuracy

def rc(X,y,alpha=1,solver='auto',random_state=1):
    from sklearn.linear_model import RidgeClassifier
    model = RidgeClassifier(alpha=alpha,solver=solver,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("RC:", accuracy)
    return accuracy

def sgd(X,y):
    from sklearn import linear_model
    model = linear_model.SGDClassifier(random_state=1)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("SGD:", accuracy)
    return accuracy

def lda(X,y,solver='svd'):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    model = LinearDiscriminantAnalysis(solver=solver)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("LDA:", accuracy)
    return accuracy

def qda(X,y):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    model = QuadraticDiscriminantAnalysis()
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("QDA:", accuracy)
    return accuracy

def svc(X,y,kernel='linear',C=1,random_state=1):
    from sklearn.svm import SVC
    model = SVC(kernel=kernel,C=C,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("SVC:", accuracy)
    return accuracy

def kn(X,y,algorithm='auto',n_neighbors=5):
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier(algorithm=algorithm,n_neighbors=n_neighbors)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("KN:", accuracy)
    return accuracy

def gp(X,y,random_state=1):
    from sklearn.gaussian_process import GaussianProcessClassifier    
    model = GaussianProcessClassifier(random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("GP:", accuracy)
    return accuracy

def nb(X,y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("NB:", accuracy)
    return accuracy

def dt(X,y,splitter='best',max_depth=None,random_state=1):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(splitter=splitter,max_depth=max_depth,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("DT:", accuracy)
    return accuracy

def rf(X,y,n_estimators=10,max_depth=None,random_state=1):
    from sklearn.ensemble import RandomForestClassifier   
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("RF:", accuracy)
    return accuracy

def et(X,y,n_estimators=10,random_state=1):
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier(n_estimators=n_estimators,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("ET:", accuracy)
    return accuracy

def gbc(X,y,n_estimators=100,learning_rate=0.1,random_state=1):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,random_state=random_state)
    scores = cross_val_score(model,X,y,cv=cv)
    accuracy = scores.mean()
    print("GBC:", accuracy)
    return accuracy


def compModels(data_X,data_y):    
    for i,x in enumerate(data_X):
        X = x[:,0:-1]
        y = x[:,-1]
        print("Loop:", i)
        svc_acc = svc(X,y,kernel='linear',C=1)
        kn_acc = kn(X,y,algorithm='ball_tree',n_neighbors=4)
        nb_acc = nb(X,y)
        dt_acc = dt(X,y,splitter='best',max_depth=2)
        gbc_acc = gbc(X,y,n_estimators=3,learning_rate=2.2)
        mapping = {svc_acc:0,kn_acc:1,nb_acc:2,dt_acc:3,gbc_acc:4}   
        arr = [svc_acc,kn_acc,nb_acc,dt_acc,gbc_acc]
        s = 0
        for j in arr:
            if(abs(j - np.max(arr)) > 0.01):
                s = s+1        
        if(s == len(arr)-1):     
            data_y.append(mapping.get(max(mapping)))
        else:
            data_y.append(-1)


def delete(X,y,z):    
    i = 0
    while i < len(y):
        if(y[i] == -1): 
            del X[i]
            del y[i]
            del z[i]
        else:
           i += 1


def data_representation(X,y,x,data_rep):
    print("Dataset Representation:", x)
    df = pd.DataFrame(X)
    n_samples = df.shape[0]
    n_features = df.shape[1]
    n_classes = set(y)
    n_classes = len(n_classes)
    c = Counter(y)
    values = np.array(list(c.values()))/n_samples 
    class_weights_min = values.min()
    class_weights_mean = values.mean()
    class_weights_max = values.max()
    mean_min = df.mean().min()
    mean_mean = df.mean().mean()
    mean_max = df.mean().max()
    t_mean_min = stats.trim_mean(X, 0.1).min()
    t_mean_mean = stats.trim_mean(X, 0.1).mean()
    t_mean_max = stats.trim_mean(X, 0.1).max()
    median_min = df.median().min()
    median_mean = df.median().mean()
    median_max = df.median().max()
    sem_min = df.sem().min()
    sem_mean = df.sem().mean()
    sem_max = df.sem().max()
    std_min = df.std().min()
    std_mean = df.std().mean()
    std_max = df.std().max()
    mad_min = df.mad().min()
    mad_mean = df.mad().mean()
    mad_max = df.mad().max()
    var_min = df.var().min()
    var_mean = df.var().mean()
    var_max = df.var().max()
    skew_min = df.skew().min()
    skew_mean = df.skew().mean()
    skew_max = df.skew().max()
    kurt_min = df.kurtosis().min()
    kurt_mean = df.kurtosis().mean()
    kurt_max = df.kurtosis().max()
    p_corr = df.corr(method='pearson')
    np.fill_diagonal(p_corr.values,0)
    p_corr_min = p_corr.min().min()
    p_corr_mean = p_corr.mean().mean()
    p_corr_max = p_corr.max().max()
    k_corr = df.corr(method='kendall')
    np.fill_diagonal(k_corr.values,0)
    k_corr_min = k_corr.min().min()
    k_corr_mean = k_corr.mean().mean()
    k_corr_max = k_corr.max().max()
    s_corr = df.corr(method='spearman')
    np.fill_diagonal(s_corr.values,0)
    s_corr_min = s_corr.min().min()
    s_corr_mean = s_corr.mean().mean()
    s_corr_max = s_corr.max().max()
    cov_min = df.cov().min().min()
    cov_mean = df.cov().mean().mean()
    cov_max = df.cov().max().max()
    variation = stats.variation(X)
    variation_min = variation.min()
    variation_mean = variation.mean()
    variation_max = variation.max()
    z_score = stats.zscore(X)
    z_score_min = z_score.min()
    z_score_mean = z_score.mean()
    z_score_max = z_score.max()
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    iqr_min = IQR.min()
    iqr_mean = IQR.mean()
    iqr_max = IQR.max()
    iqr_mul_outliers_sum = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1).sum()
    iqr_mul_outliers_per = iqr_mul_outliers_sum/n_samples
    iqr_uni_outliers_sum = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum().sum()
    iqr_uni_outliers_per = iqr_uni_outliers_sum/(n_samples*n_features)
    z = np.abs(stats.zscore(X))
    z = pd.DataFrame(z)
    z_mul_outliers_sum = (z>3).any(axis=1).sum()
    z_mul_outliers_per = z_mul_outliers_sum/n_samples 
    z_uni_outliers_sum = (z>3).sum().sum()
    z_uni_outliers_per = z_uni_outliers_sum/(n_samples*n_features)
    X_entr_min = entropy(df).min()
    X_entr_mean = entropy(df).mean()
    X_entr_max = entropy(df).max()
    y_entr = entropy(y)
    mutual_info_min = mutual_info_classif(X,y,random_state=1).min()
    mutual_info_mean = mutual_info_classif(X,y,random_state=1).mean()
    mutual_info_max = mutual_info_classif(X,y,random_state=1).max()
    if(mutual_info_mean != 0):    
        en = y_entr/mutual_info_mean
        ns = (X_entr_mean-mutual_info_mean)/mutual_info_mean
    else:
        en = 0
        ns = 0  
    clf = IsolationForest(behaviour="new",contamination='auto',random_state=1)
    if_anomalies = clf.fit_predict(X)
    if_an = np.where(if_anomalies == -1)
    if_an_sum = len(if_an[0])
    if_an_per = if_an_sum/n_samples
    clf = LocalOutlierFactor(contamination='auto')
    lof_anomalies = clf.fit_predict(X)
    lof_an = np.where(lof_anomalies == -1)
    lof_an_sum = len(lof_an[0])
    lof_an_per = lof_an_sum/n_samples
    clf=svm.OneClassSVM(gamma='scale')
    svm_anomalies=clf.fit_predict(X)
    svm_an = np.where(svm_anomalies == -1)
    svm_an_sum = len(svm_an[0])
    svm_an_per = svm_an_sum/n_samples
    pca = PCA(n_components=2,random_state=1)
    X_pca = pca.fit_transform(X)
    pca_ev_sum = pca.explained_variance_ratio_.sum()
    pca_ev_min = pca.explained_variance_ratio_.min()
    pca_ev_mean = pca.explained_variance_ratio_.mean()
    pca_ev_max = pca.explained_variance_ratio_.max()
    pca_sv_sum = pca.singular_values_.sum()
    pca_sv_min = pca.singular_values_.min()
    pca_sv_mean = pca.singular_values_.mean()
    pca_sv_max = pca.singular_values_.max()
    pca_nv = pca.noise_variance_
    tsvd = TruncatedSVD(n_components=2,random_state=1)
    tsvd.fit_transform(X)
    tsvd_ev_sum = tsvd.explained_variance_ratio_.sum()
    tsvd_ev_min = tsvd.explained_variance_ratio_.min()
    tsvd_ev_mean = tsvd.explained_variance_ratio_.mean()
    tsvd_ev_max = tsvd.explained_variance_ratio_.max()
    tsvd_sv_sum = tsvd.singular_values_.sum()
    tsvd_sv_min = tsvd.singular_values_.min()
    tsvd_sv_mean = tsvd.singular_values_.mean()
    tsvd_sv_max = tsvd.singular_values_.max()
    anova = f_classif(X,y) 
    anova_f_min = anova[0].min()
    anova_f_mean = anova[0].mean()
    anova_f_max = anova[0].max()
    anova_sum = len(anova[1][anova[1]<0.05])
    anova_per = len(anova[1][anova[1]<0.05])/n_features
    U, singv, VT = svd(X)
    singv_min = singv.min()
    singv_mean = singv.mean()
    singv_max = singv.max()
    bestfeatures = SelectKBest(score_func=chi2,k=2)
    fit = bestfeatures.fit(X,y)
    dfscores = fit.scores_
    chi2_based_scores_min = dfscores.min()
    chi2_based_scores_mean = dfscores.mean()
    chi2_based_scores_max = dfscores.max()
    estimator = LogisticRegression(random_state=1)
    selector = RFECV(estimator, step=1, cv=5)
    og_X = pd.DataFrame(X)
    new_X = selector.fit_transform(og_X, y)
    rfecv_per_optimal_feat = selector.n_features_/n_features
    dt_best = DecisionTreeClassifier(max_depth=1,splitter='best',random_state=1)
    scores = cross_val_score(dt_best,X,y,cv=5)
    dt_best_acc = scores.mean()
    dt_rnd = DecisionTreeClassifier(max_depth=1,splitter='random',random_state=1)
    scores = cross_val_score(dt_rnd,X,y,cv=5)
    dt_rnd_acc = scores.mean()
    kn1 = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(kn1,X,y,cv=5)
    kn1_acc = scores.mean()
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda,X,y,cv=5)
    lda_acc = scores.mean()
    nb = GaussianNB()
    scores = cross_val_score(nb,X,y,cv=5)
    nb_acc = scores.mean()
    tmp = [n_samples,n_features,n_classes,class_weights_min,class_weights_mean,class_weights_max,
           mean_min,mean_mean,mean_max,t_mean_min,t_mean_mean,t_mean_max,median_min,median_mean,median_max,
           sem_min,sem_mean,sem_max,std_min,std_mean,std_max,mad_min,mad_mean,mad_max,var_min,var_mean,var_max,
           skew_min,skew_mean,skew_max,kurt_min,kurt_mean,kurt_max,p_corr_min,p_corr_mean,p_corr_max,
           k_corr_min,k_corr_mean,k_corr_max,s_corr_min,s_corr_mean,s_corr_max,cov_min,cov_mean,cov_max,
           variation_min,variation_mean,variation_max,z_score_min,z_score_mean,z_score_max,iqr_min,iqr_mean,iqr_max,
           iqr_mul_outliers_sum,iqr_mul_outliers_per,iqr_uni_outliers_sum,iqr_uni_outliers_per,z_mul_outliers_sum,z_mul_outliers_per,z_uni_outliers_sum,z_uni_outliers_per,
           X_entr_min,X_entr_mean,X_entr_max,y_entr,mutual_info_min,mutual_info_mean,mutual_info_max,en,ns,
           if_an_sum,if_an_per,lof_an_sum,lof_an_per,svm_an_sum,svm_an_per,
           pca_ev_sum,pca_ev_min,pca_ev_mean,pca_ev_max,pca_sv_sum,pca_sv_min,pca_sv_mean,pca_sv_max,pca_nv,
           tsvd_ev_sum,tsvd_ev_min,tsvd_ev_mean,tsvd_ev_max,tsvd_sv_sum,tsvd_sv_min,tsvd_sv_mean,tsvd_sv_max,anova_f_min,anova_f_mean,anova_f_max,anova_sum,anova_per,
           singv_min,singv_mean,singv_max,
           chi2_based_scores_min,chi2_based_scores_mean,chi2_based_scores_max,rfecv_per_optimal_feat,
           dt_best_acc,dt_rnd_acc,kn1_acc,lda_acc,nb_acc]
    tmp = np.asarray(tmp)
    tmp = tmp.reshape(1,len(tmp))
    data_rep.append(tmp)