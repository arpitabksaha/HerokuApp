from flask import Flask,request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import simplejson as json
from scipy import stats
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau, pearsonr, spearmanr
import math
import statsmodels.api as sm
import seaborn as sns
import statistics

app = Flask(__name__)

#Notes - 3)Add bar chart 4) Build heatmap from cor_data
#variable               variable2  correlation correlation_label(imp)
#1   ClosePrice                 Acreage    -0.101379             -0.10



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
from mpld3 import _display
_display.NumpyEncoder = NumpyEncoder


@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        summary = df.describe().transpose()
        print(df.columns)
        df.columns = df.columns.str.replace(' ', '')
        df_clean = df.copy()

        #Remove Outliers
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1

        df_clean = df_clean[~((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)]
        df_clean = df_clean.dropna(axis=1, how='all')
        #print(df_clean.columns)
        #print(cat_df)

        # calculating percentage of sales with seller concessions
        percentageofSaleswithSellercount = df.apply(lambda x: True if x['SellerConcessionYN'] == True else False,axis=1)
        PercentageofDSaleswithSellerConcessions = round(100 * (len(percentageofSaleswithSellercount[percentageofSaleswithSellercount == True].index)) / (len(df.index) - 1),2)

        # calculating average seller concessions amt and %
        SellerConcession = df[df['SellerConcessionYN'] == True]
        AverageSellerConcession = SellerConcession['SellerConcessionAmount']
        AverageSellerConcessionRatio = SellerConcession['SellerConcessionAmount'] / SellerConcession['ClosePrice']
        AverageSellerConcessionAmount = AverageSellerConcession.mean(axis=0, skipna=True)
        AverageSellerConcessionAmount = round(AverageSellerConcessionAmount,2)
        AverageSellerConcessionPercent = AverageSellerConcessionRatio.mean(axis=0, skipna=True)
        AverageSellerConcessionPercent = round(AverageSellerConcessionPercent*100,2)

        # calculating median seller concessions
        MedianSellerConcessionAmount = round(AverageSellerConcession.median(axis=0, skipna=True),2)

        # calculating statistics - Pvalue

        def pearsonr_pval(x, y):
            return pearsonr(x, y)[1]

        cor_data_pval = (df.drop(columns=['WaterFrontageFeet']).corr(method=pearsonr_pval).stack().reset_index().rename(
            columns={0: 'pvalues', 'level_0': 'variable', 'level_1': 'variable2'}))
        #print(cor_data_pval)
        cor_data_pval = cor_data_pval[cor_data_pval['variable'] == 'ClosePrice']
        #print("check2")
        cor_data_pval = cor_data_pval[cor_data_pval['variable2'] != 'ClosePrice']
        #print(cor_data_pval)
        cor_data_pval.assign(SignificantYN='NA')
        cor_data_pval.loc[(cor_data_pval.pvalues <= 0.05), 'SignificantYN'] = 'Yes'
        cor_data_pval.loc[(cor_data_pval.pvalues > 0.05), 'SignificantYN'] = 'No'

        # calculating statistics - Pearson's r value

        cor_data = (df.drop(columns=['WaterFrontageFeet']).corr(method='pearson').stack().reset_index().rename(
            columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
        cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
        cor_data = cor_data[cor_data['variable'] == 'ClosePrice']
        cor_data = cor_data[cor_data['variable2'] != 'ClosePrice']
        cor_datatable = cor_data.copy()
        #print(cor_data)
        cor_datatable.rename(columns={'variable2': 'Dimension', 'correlation_label': 'Correlation'}, inplace=True)
        #print("cor_data")
        #print(cor_data)
        df_suffix = pd.merge(cor_data_pval, cor_datatable, left_on='variable2', right_on='Dimension', how='outer',
                             suffixes=('_left', '_right'))
        df_suffix_p = df_suffix[['Dimension', 'pvalues', 'Correlation', 'SignificantYN']]
        cor_datatable_sig = df_suffix[['Dimension', 'Correlation', 'SignificantYN']]

        cor_datatable_sig.rename(columns={'Dimension': 'Feature', 'Correlation': 'Correlation with Closing Price',
                                          'SignificantYN': 'Significant(Y/N)'}, inplace=True)
        #Calculate feature range table

        ##################################################Function to calculate coeff. of linear regression

        def calc_slope(df, feature):
            df = df.fillna(0)
            X = df[feature].values.reshape(-1, 1)
            Y = df['ClosePrice'].values.reshape(-1, 1)
            regressor = LinearRegression()
            regressor.fit(X, Y)
            lrcoeff = math.ceil(regressor.coef_)
            listeval = [feature, lrcoeff]
            #print(listeval)
            return listeval

        ##################################################Function to calculate confidence interval from stackoverflow
        def r_to_z(r):
            return math.log((1 + r) / (1 - r)) / 2.0

        def z_to_r(z):
            e = math.exp(2 * z)
            return ((e - 1) / (e + 1))

        def r_confidence_interval(r, data):
            alpha = 0.05
            n = len(data.index) + 1
            z = r_to_z(r)
            se = 1.0 / math.sqrt(n - 3)
            z_crit = stats.norm.ppf(1 - alpha / 2)  # 2-tailed z critical value

            lo = z - z_crit * se
            hi = z + z_crit * se

            r_lo = z_to_r(lo) * 100
            r_hi = z_to_r(hi) * 100
            inter = [math.ceil(r_lo), math.ceil(r_hi)]
            return (inter)

        # print(r_confidence_interval(0.66))
        ##################################### calculate pearson's correlation values

        cor_datatable_sig = cor_datatable_sig[cor_datatable_sig['Significant(Y/N)'] == 'Yes']
        cor_datatable_sig = cor_datatable_sig[['Feature', 'Correlation with Closing Price']]
        cor_datatable_sig = cor_datatable_sig.sort_values('Correlation with Closing Price', ascending=False)
        cor_datatable_sig.assign(Rangecorr='Default')
        interval = []
        for i in cor_datatable_sig['Correlation with Closing Price']:
            interval.append('  -  '.join(str(elem) for elem in (r_confidence_interval(float(i), df))))
        cor_datatable_sig['Rangecorr'] = interval
        cor_datatable_sig = cor_datatable_sig[['Feature', 'Rangecorr']]

        # slope = pd.DataFrame(columns=["Feature","Slope"])
        slope_df = pd.DataFrame(columns=['Features', 'Slope'])
        for i in cor_datatable_sig['Feature']:
            slope_df = slope_df.append(pd.DataFrame([calc_slope(df, i)], columns=slope_df.columns))
        cor_datatable_sig.reset_index(inplace=True)
        slope_df.reset_index(inplace=True)
        Combined_table = cor_datatable_sig.join(slope_df, how='inner', lsuffix='Features', rsuffix='Features')
        Combined_table[['LCI', 'UCI']] = Combined_table.Rangecorr.str.split(" - ", expand=True)
        Combined_table['LCI'] = pd.to_numeric(Combined_table['LCI'])
        Combined_table['UCI'] = pd.to_numeric(Combined_table['UCI'])
        Combined_table['Slope'] = pd.to_numeric(Combined_table['Slope'])
        Combined_table['LCI'] = Combined_table['LCI'] * Combined_table['Slope'] / 100
        Combined_table['UCI'] = Combined_table['UCI'] * Combined_table['Slope'] / 100
        Combined_table['LCI'] = Combined_table['LCI'].map(lambda a: math.ceil(a))
        Combined_table['UCI'] = Combined_table['UCI'].map(lambda a: math.ceil(a))
        Combined_table['LCI'] = Combined_table['LCI'].apply(str)
        Combined_table['UCI'] = Combined_table['UCI'].apply(str)
        Combined_table['Adjusted Tweak Amount'] = Combined_table['LCI'].str.cat(Combined_table['UCI'], sep="-")
        cor_datatable_sig.rename(columns={'Rangecorr': 'Tweak Amount'}, inplace=True)
        Combined_table = Combined_table[['Features', 'Adjusted Tweak Amount']]
        Combined_table['Adjusted Tweak Amount'] = '$' + Combined_table['Adjusted Tweak Amount'].astype(str)
        Combined_table.set_index('Features', inplace=True)

        #print(Combined_table)
        #cor_data.set_index('variable',inplace=True)

        fig1, ax1 = plt.subplots()
        x1 = df_clean.EstFinAbvGrdSqFt
        #x1CI = df_clean.EstFinAbvGrdSqFt.to_numpy()
        #y1CI = df_clean.ClosePrice.to_numpy()
        y1 = df_clean.ClosePrice
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)
        sns.regplot(x="EstFinAbvGrdSqFt", y="ClosePrice", data=df_clean,ax=ax1,truncate=True, scatter_kws={"color": "#1A5276"}, line_kws={"color": "red","alpha":1});

        #est = sm.OLS(y1CI, x1CI)
        #est2 = est.fit()
        #ci1 = est2.conf_int(alpha=0.05, cols=None)
        #print(x1.to_numpy())
        #scatter = ax1.scatter(x1,y1)
        line1 = slope1 * x1 + intercept1
        #plt.plot(x1, line1, 'r', label='y={:.2f}x+{:.2f}'.format(slope1, intercept1))
        #ax1.legend()
        #ax1.fill_between(x1CI, (y1CI - ci1[0][0]), (y1CI + ci1[0][1]), color='b', alpha=.1)
        ax1.grid(color='lightgrey', linestyle='solid')
        ax1.set_xlabel("Reported Living Area(SF)", fontsize=15)
        ax1.set_ylabel("Reported Sales Price($)", fontsize=15)
        ax1.set_title("Scatter Plot for Reported Living Area and Reported Sales Price", size=20)
        #sns.lineplot(data=df_clean, x="EstFinAbvGrdSqFt", y="ClosePrice", ax=ax1)
        json01 = json.dumps(mpld3.fig_to_dict(fig1))

        fig2, ax2 = plt.subplots()
        x2 = df_clean.Acreage
        y2 = df_clean.ClosePrice
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)
        sns.regplot(x="Acreage", y="ClosePrice", data=df_clean,ax=ax2,truncate=True, scatter_kws={"color": "#1A5276"}, line_kws={"color": "red","alpha":1});


        #scatter2 = ax2.scatter(x2, y2)
        #line2 = slope2 * x2 + intercept2
        #plt.plot(x2, line2, 'r', label='y={:.2f}x+{:.2f}'.format(slope2, intercept2))
        ax2.legend()
        ax2.grid(color='lightgrey', linestyle='solid')
        ax2.set_xlabel("Reported Site Size (ac)", fontsize=15)
        ax2.set_ylabel("Reported Sales Price($)", fontsize=15)
        ax2.set_title("Scatter Plot for Reported Site Size (ac) and Reported Sales Price", size=20)
        json02 = json.dumps(mpld3.fig_to_dict(fig2))

        fig3, ax3 = plt.subplots()
        x3 = df_clean.YearBuilt
        y3 = df_clean.ClosePrice
        slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(x3, y3)
        sns.regplot(x="YearBuilt", y="ClosePrice", data=df_clean, ax=ax3,truncate=True, scatter_kws={"color": "#1A5276"}, line_kws={"color": "red","alpha":1});


        #scatter3 = ax3.scatter(x3, y3)
        #line3 = slope3 * x3 + intercept3
        #plt.plot(x3, line3, 'r', label='y={:.2f}x+{:.2f}'.format(slope3, intercept3))
        ax3.legend()
        ax3.grid(color='lightgrey', linestyle='solid')
        ax3.set_xlabel("Year Built", fontsize=15)
        ax3.set_ylabel("Reported Sales Price($)", fontsize=15)
        ax3.set_title("Scatter Plot for Seller Concession Amount and Reported Sales Price", size=20)
        json03 = json.dumps(mpld3.fig_to_dict(fig3))

        fig4, ax4 = plt.subplots()
        x4 = df_clean.YearRemodeled
        y4 = df_clean.ClosePrice
        slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(x4, y4)
        sns.regplot(x="YearRemodeled", y="ClosePrice", data=df_clean, ax=ax4,truncate=True, scatter_kws={"color": "#1A5276"}, line_kws={"color": "red","alpha":1});

        #scatter4 = ax4.scatter(x4, y4)
        #line4 = slope4 * x4 + intercept4
        #plt.plot(x4, line4, 'r', label='y={:.2f}x+{:.2f}'.format(slope4, intercept4))
        ax4.legend()
        ax4.grid(color='lightgrey', linestyle='solid')
        ax4.set_xlabel("Year Remodeled", fontsize=15)
        ax4.set_ylabel("Reported Sales Price($)", fontsize=15)
        ax4.set_title("Scatter Plot for Year Remodeled and Reported Sales Price", size=20)
        json04 = json.dumps(mpld3.fig_to_dict(fig4))


        fig_bar1, ax_bar1 = plt.subplots()
        keys1 = list(df_clean.BathsFull)
        values1 = list(df_clean.ClosePrice)
        ax_bar1 = plt.bar(keys1, values1, width=0.5)
        #mean_bar1 = df_clean['ClosePrice'].mean()
        #ax_bar1.axvline(mean_bar1, color='r', linestyle='-')
        #print(df_clean['BathsFull'])
        plt.xticks([0.0,1.0,2.0,3.0,4.0])
        plt.xlabel("Number of Full Bathrooms")
        plt.ylabel("Reported Sales Price($)")
        fig_bar1 = ax_bar1[0].figure
        json_bar1 = json.dumps(mpld3.fig_to_dict(fig_bar1))
        #bar_chart = mpld3.fig_to_html(fig_bar1)
        #print(df_clean.dtypes)
        #sns.barplot(x='BathsHalf', y='ClosePrice', data=df_clean, ax=ax_bar1)
        #ax_bar1 = df_clean.plot.bar(x='BathsHalf', y='ClosePrice' )
        #json_bar1 = json.dumps(mpld3.fig_to_dict(fig_bar1))

        fig_bar2, ax_bar2 = plt.subplots()
        keys2 = list(df_clean.BathsHalf)
        values2 = list(df_clean.ClosePrice)
        ax_bar2 = plt.bar(keys2, values2, width=0.5)
        plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0])
        plt.xlabel("Number of Half Bathrooms")
        plt.ylabel("Reported Sales Price($)")
        fig_bar2 = ax_bar2[0].figure
        json_bar2 = json.dumps(mpld3.fig_to_dict(fig_bar2))

        fig_bar3, ax_bar3 = plt.subplots()
        keys3 = list(df_clean.BedsTotal)
        values3 = list(df_clean.ClosePrice)
        ax_bar3 = plt.bar(keys3, values3, width=0.5)
        plt.xticks([0.0, 1.0, 2.0, 3.0, 4.0])
        plt.xlabel("Number of Bedrooms")
        plt.ylabel("Reported Sales Price($)")
        fig_bar3 = ax_bar3[0].figure
        json_bar3 = json.dumps(mpld3.fig_to_dict(fig_bar3))

        # Histogram plot
        fig_h1, ax_h1 = plt.subplots()
        mean = df_clean['ClosePrice'].mean()
        ax_h1 = sns.distplot(df_clean.ClosePrice)
        ax_h1.axvline(mean, color='r', linestyle='-')
        json_h1 = json.dumps(mpld3.fig_to_dict(fig_h1))

        fig_h2, ax_h2 = plt.subplots()
        mean2 = df_clean['EstFinAbvGrdSqFt'].mean()
        ax_h2 = sns.distplot(df_clean.EstFinAbvGrdSqFt)
        ax_h2.axvline(mean2, color='r', linestyle='-')
        json_h2 = json.dumps(mpld3.fig_to_dict(fig_h2))

        #heatmap
        cor_data = cor_data[['variable','variable2','correlation_label']]
        #cor_data.set_index('variable',inplace=True)
        cor_data['variable'] = cor_data['variable'].astype('str')
        cor_data['variable2'] = cor_data['variable2'].astype('str')
        cor_data['correlation_label'] = cor_data['correlation_label'].astype('float')
        result = cor_data.pivot(index='variable', columns='variable2', values='correlation_label')
        column_labels = list(cor_data.variable2)
        row_labels = ['Closing Price']
        print(result.values)
        fig_hm, ax_hm = plt.subplots()
        im = ax_hm.imshow(result.values)
        #ax_hm.set_xticklabels(column_labels)
        #ax_hm.set_yticklabels(row_labels)
        plt.setp(ax_hm.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        fig_hm.tight_layout()

        print((np.array(result.values[0]))[0])
        print(result.values[0,0])




        ax_hm.pcolor(result.values, cmap=plt.cm.Reds)
        #ax_hm.set_xticks(np.arange(result.shape[0]) + 0.5, minor=False)
        #ax_hm.set_yticks(np.arange(result.shape[1]) + 0.5, minor=False)
        #ax_hm.set_frame_on(False)
        #ax_hm.axes.get_yaxis().set_visible(False)
        #fig_hm.tight_layout()


        #ax_hm.set_xticklabels(column_labels, minor=False)
        #ax_hm.set_yticklabels(row_labels, minor=False)
        ax_hm.invert_yaxis()

        json_hm = json.dumps(mpld3.fig_to_dict(fig_hm))
        #plt.show()




        #summary.to_html(classes='male')

        return render_template('upload.html', tables=[Combined_table.to_html(classes='male')], json01=json01, json02=json02, json03=json03, json04=json04, json_h1=json_h1,json_h2=json_h2,json_bar1=json_bar1,json_bar2=json_bar2,json_bar3=json_bar3,json_hm=json_hm, PercentageofDSaleswithSellerConcessions=PercentageofDSaleswithSellerConcessions, AverageSellerConcessionAmount=AverageSellerConcessionAmount,AverageSellerConcessionPercent=AverageSellerConcessionPercent,MedianSellerConcessionAmount=MedianSellerConcessionAmount)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
