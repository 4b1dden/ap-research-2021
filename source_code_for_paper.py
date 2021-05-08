# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


# DECLARATIONS
pss_cols = ['pss_traffic_high_density', 'pss_traffic_high_speed', 'pss_rail_infrastructure', 'pss_confusing_xsections', 'pss_poorly_lit', 'pss_road_bike_lanes', 'pss_separate_bike_lanes', 'pss_pavements', 'pss_entrances_exits', 'pss_one_way_street', 'pss_poor_road_conditions', 'pss_winter', 'pss_wet_roads', 'pss_parked_cars']
rt_cols = ['rt_rule_breaking', 'rt_red_light_run', 'rt_risk_taking_alone', 'rt_rule_breaking_relative_risk', 'rt_alcohol', 'rt_rule_obeying_difficulty', 'rt_front_light', 'rt_back_light', 'rt_reflexive_apparell', 'rt_road_to_pavement_switch', 'rt_pavement_to_road_switch', 'rt_prefers_road', 'rt_subjective_faster_riding', 'rt_subjective_safer_riding', 'rt_hand_signals']
hw_cols = ['hw_urban_env', 'hw_rural_env', 'hw_comfort', 'hw_financial_investment', 'hw_practical_reasons', 'hw_comfort_reasons', 'hw_aesthetic', 'hw_weather_influence']
prefixes = ["pss", "rt", "hw"]
output_cols = ['name', 'mean', 'median']

models = {
    "pss": {
        'traffic': ['pss_traffic_high_density', 'pss_traffic_high_speed'],
        'conditions': ["pss_poor_road_conditions", "pss_winter", "pss_wet_roads"],
        'dangerous': ['pss_traffic_high_density', 'pss_traffic_high_speed', 'pss_rail_infrastructure', 'pss_confusing_xsections', 'pss_poorly_lit', 'pss_entrances_exits', 'pss_one_way_street', 'pss_parked_cars']
    },
    "rt": {
        'reckless': ['rt_rule_breaking', 'rt_red_light_run', 'rt_risk_taking_alone'],
        'reckless_2': ['rt_red_light_run', 'rt_alcohol'],
        'prefers_road': ['rt_prefers_road', 'rt_pavement_to_road_switch'],
        'perceived_faster': ['rt_subjective_faster_riding'],
        'perceived_safer': ['rt_subjective_safer_riding'],
        'equipment': ['rt_front_light', 'rt_back_light']
    },
    "hw": {
        "urban": ['hw_urban_env'],
    }
}

# METHODS

# computes single variable analyses as seen in Appendix 5
def compute_single_var_analyses(df, target_cols, general_stats_prefix_list, output_cols):
    single_field_analysis = pd.DataFrame(columns=output_cols)
    for col_name in target_cols:
        single_field_analysis = single_field_analysis.append(
            pd.Series([col_name, df[col_name].mean(), df[col_name].median()], index=output_cols),
            ignore_index=True
        )
        
    for prefix in general_stats_prefix_list:
        mask = single_field_analysis['name'].str.startswith(prefix)
        mean_ordered = single_field_analysis[mask].sort_values('mean')
        median_ordered = single_field_analysis[mask].sort_values('median')
        # print(mean_ordered)
        # print(median_ordered)
        
    return single_field_analysis

# computes means for every custom model
def compute_stats(df, general_stats_prefix_list, models_map):
    for prefix in general_stats_prefix_list:
        df = df.drop(columns=[prefix+"_mean"], errors='ignore')
        idx = df.columns.str.startswith(prefix)
        
        df[prefix+"_mean"] = df.iloc[:,idx].mean(axis=1)
        
    for category,category_models in models_map.items():
        for model_name, field_list in category_models.items():
            final_col_name = category + "_" + model_name
            
            df = df.drop(columns=[final_col_name+"_mean"], errors='ignore')
            df[final_col_name+"_mean"] = df[field_list].mean(axis=1)

    return df

# used to render pairgrids, such as those in Appendix 4
def render_pairgrid(to_plot_df, title, optional_vars=[]):
    v = None if len(optional_vars) == 0 else optional_vars
    
    d = sns.PairGrid(to_plot_df, vars=v)
    d = d.map_upper(plt.scatter, marker='+')
    d = d.map_lower(sns.kdeplot, cmap="hot",shade=True)
    d = d.map_diag(sns.kdeplot, shade=True)

    d.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
    d.fig.suptitle(title + " (" + str(to_plot_df.shape[0]) + " data points)")

    plt.show()


# plot linear regression, df = dataframe with the data, x {string} = key for x axis data, y {string} = key for y axis data 
def plt_lin_reg(df, x, y):
    slope, intercept, r_value, pv, se = stats.linregress(df[x],df[y])

    sns.regplot(x=x, y=y, data=df, label="y={0:.1f}x+{1:.1f}".format(slope, intercept)).legend(loc="best")
    print("r^2=", r_value**2)
    print("p value=", pv)
    print("slope=", slope)
    print('intercept', intercept)
    
# used to filter entire dataset to a dataset with urban cyclists only 
def urban_only(df):
    return df[(df['bike_env'] == 'mestskom') | (df['bike_env'] == 'nedokážem zhodnotiť')]

#used prec = 100
def lasso_optimize_r_2_through_lambda(prec, X_train, y_train, X_test, y_test):
    lambdas, test_r_squares = [], []
    
    for i in range(prec):
        _lambda = i / prec
        lambdas.append(_lambda)
        model_lasso = linear_model.Lasso(alpha=_lambda)
        model_lasso.fit(X_train, y_train) 
        pred_train_lasso= model_lasso.predict(X_train)

        pred_test_lasso= model_lasso.predict(X_test)
        test_r_square = r2_score(y_test, pred_test_lasso)
        test_r_squares.append(test_r_square)

        sns.lineplot(x=lambdas, y=test_r_squares)

        max_y = max(test_r_squares)  
        max_x = lambdas[test_r_squares.index(max_y)]
       
        return max_y, max_x

# computes all custom models
def get_all_custom_models(model_map):
    custom_models = []
    for category,category_models in model_map.items():
        for model_name, field_list in category_models.items():
            custom_models.append(category + "_" + model_name+"_mean")
            
    return custom_models

# EXAMPLES

plt_lin_reg(df, "rt_mean", "hw_mean")
plt_lin_reg(stats_df, "rt_reflexive_apparell", "hw_urban_env")
plt_lin_reg(urban_only(stats_df), "rt_reflexive_apparell", "hw_urban_env")

render_pairgrid(urban_only(stats_df), "Urban cycling population only and primitive reduction.", ['pss_mean', 'rt_mean', 'hw_mean', 'hw_urban_env'])