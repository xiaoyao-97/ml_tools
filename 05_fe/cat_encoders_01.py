"""one hot
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[cat_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_cols))
df_encoded = pd.concat([df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(cat_cols, axis=1)"""

"""01: James Stein Encoder
from category_encoders.james_stein import JamesSteinEncoder
JSE_encoder = JamesSteinEncoder()
train_jse = JSE_encoder.fit_transform(train[feature_list], target)
test_jse = JSE_encoder.transform(test[feature_list])
"""

"""02: Helmert Encoder
from category_encoders.helmert import HelmertEncoder
HE_encoder = HelmertEncoder(feature_list)
train_he = HE_encoder.fit_transform(train[feature_list], target)
test_he = HE_encoder.transform(test[feature_list])
"""

"""03: CatBoost Encoder
from category_encoders.cat_boost import CatBoostEncoder
CBE_encoder = CatBoostEncoder()
train_cbe = CBE_encoder.fit_transform(train[feature_list], target)
test_cbe = CBE_encoder.transform(test[feature_list])
"""

"""04: Weight of Evidence Encoder
from category_encoders.woe import WOEEncoder
WOE_encoder = WOEEncoder()
train_woe = WOE_encoder.fit_transform(train[feature_list], target)
test_woe = WOE_encoder.transform(test[feature_list])
"""

"""from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder"""