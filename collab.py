#   ---------------------------------  Library    ---------------------------------
import pandas as pd
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
#   Plot Graph
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#   Content-based recommendation libraries
from scipy.spatial.distance import euclidean, hamming, cosine, cdist
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
#   Collab
import scipy
from sklearn.metrics.pairwise import cosine_similarity
#   Ultility
import time
import streamlit as st
from random import sample

#   Global config
st.set_page_config(page_title='CookingNaNa',page_icon='NaNaLogo.png',layout='centered',initial_sidebar_state='expanded')

st.title(" Cooking NaNa Recipe Recommendation ")

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


#   ---------------------------------Global function---------------------------------
recipe_list=[]  #   Global variables

#   print image and recipe name
def get_image_url(recipe_list):
    recipe_list=[recipe_list]
    return df_raw_recipe[df_raw_recipe.index.isin(recipe_list)]['image_url'].values[0]
def get_recipe_NameList(recipe_list):
    recipe_list=[recipe_list]
    return df_raw_recipe[df_raw_recipe.index.isin(recipe_list)]['recipe_name'].to_list()
    
# text processing
def token(words):
    words = re.sub("[^a-zA-Z]"," ",words)
    text = words.lower().split()
    return " ".join(text)

stop_words = stopwords.words('english')+['m','h','u','directions', 'f']
def stopwords(review):
    text = [ word.lower() for word in review.split() if word.lower() not in stop_words]
    return " ".join(text)

lem = WordNetLemmatizer()
def lemma(text):
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return lem_text

def update_matching_list(search_word):
    name_lst = lemma(stopwords(token(search_word)))
    st.write('search_lst:    ', name_lst)
    df_raw_recipe['matching'] = df_raw_recipe['recipe_name_clean'].apply(lambda x: sum(
     [1 if lem.lemmatize(a) in name_lst else 0 for a in x.split(" ")])) 

def to_join_ingredients(x):
    if type(x)==list:
        return ', '.join(x)
    return x


#   ---------------------------------Load CSV---------------------------------
@st.cache(persist=True,allow_output_mutation=True)

def get_recipe_data():
    url_recipe = "data_recipe_deploy.csv"    #data_recipe_clean  data_recipe_deploy
    df = pd.read_csv(url_recipe,index_col='recipe_id')
    df['ingredients'] = df['ingredients'].str.split('^')
    df['ingredients'] = df['ingredients'].apply(lambda x: to_join_ingredients(x))
    return df
df_raw_recipe = get_recipe_data()

@st.cache(persist=True,allow_output_mutation=True)

def get_rating_data():
    url_rating = "data_rating_clean.csv"
    return pd.read_csv(url_rating)
df_raw_rating = get_rating_data()
    
@st.cache(persist=True,allow_output_mutation=True)

def recommender_hot_recipe_data():
    url = "hot_recipe_by_rating.csv"
    return pd.read_csv(url)

#   ---------------------------------  Function ---------------------------------
def recommender_hot_recipe(n=3):
    df_rating_count = recommender_hot_recipe_data()
    hot_recipe_lst = df_rating_count['recipe_id'].tolist()
    random_hot_recipe_lst = sample(hot_recipe_lst,n)
    return random_hot_recipe_lst

def recommender_recipe_name(search_word, N=3):
    update_matching_list(search_word)
    TopNRecommendation =  df_raw_recipe.sort_values(['matching','aver_rate'],ascending=False).head(N)
    TopNRecommendation_lst = TopNRecommendation.index.tolist()
    return TopNRecommendation_lst

def recommender_nutrition(nutrition_input, search_word, N = 5):
    update_matching_list(search_word)
    if search_word == token(def_search_bar.lower()):
        nutrition_input['matching']=0
        df_raw_recipe['matching']=0
    else:
        nutrition_input['matching']=len(lemma(stopwords(token(search_word))))

    df_score = pd.DataFrame(list(df_raw_recipe.index),columns=['recipe_id'])
    hamming_col = ['niacin_Scaled', 'sugars_Scaled', 'sodium_Scaled','carbohydrates_Scaled', 'vitaminB6_Scaled', 'calories_Scaled',\
            'thiamin_Scaled', 'fat_Scaled', 'folate_Scaled','caloriesFromFat_Scaled', 'calcium_Scaled','fiber_Scaled','magnesium_Scaled',\
            'iron_Scaled', 'cholesterol_Scaled','protein_Scaled', 'vitaminA_Scaled', 'potassium_Scaled','saturatedFat_Scaled', 'Total_Time_Normalized','matching']
    df_score['Distance'] = cdist( np.reshape(list(nutrition_input.values()) , (1,21)) , df_raw_recipe.loc[:,hamming_col],'euclidean').T
    df_score = df_score.merge(df_raw_recipe[['aver_rate']], on ='recipe_id')
    df_score.set_index('recipe_id',inplace=True)
    df_score.sort_values(["Distance","aver_rate"],ascending=[True,False],inplace=True)
    recommend_lst = list(df_score.index[0:N])
    return recommend_lst

def recommender_CookingMethod(cooking_method_input, search_word, N = 5):
    update_matching_list(search_word)
    cm_arr = [0,0,0,0,0,0,0,0,0,0]
    if search_word == token(def_search_bar.lower()):
        cm_arr.append(0)
        df_raw_recipe['matching']=0
    else:
        cm_arr.append(len(lemma(stopwords(token(search_word)))))

    for x in range(10):
        if cooking_options[x] == cooking_method_input:
            cm_arr[x] = 1

    allRecipes = pd.DataFrame(list(df_raw_recipe.index),columns=['recipe_id'])
    hamming_col = cooking_options
    hamming_col.append('matching')
    allRecipes['Distance'] = cdist( np.reshape(cm_arr , (1,len(cm_arr))) , df_raw_recipe.loc[:,hamming_col],'euclidean').T
    allRecipes = allRecipes.merge(df_raw_recipe[['aver_rate']], on ='recipe_id')
    allRecipes.set_index('recipe_id',inplace=True)
    allRecipes.sort_values(["Distance","aver_rate"],ascending=[True,False],inplace=True)
    recommend_lst = list(allRecipes.index[0:N])
    return recommend_lst

def recommender_CookingTime(search_word, cooking_time_limit, N = 5):
    name_lst = lemma(stopwords(token(search_word)))

    # create dataframe used to store the number of matching word
    allName = pd.DataFrame(df_raw_recipe.index).set_index('recipe_id')
    allName['recipe_name'] = df_raw_recipe[['recipe_name']]
    allName['aver_rate'] = df_raw_recipe[['aver_rate']]
    allName['recipe_name_clean'] = df_raw_recipe[['recipe_name_clean']]
    allName['Total_Time'] = df_raw_recipe[['Total_Time']]
    if cooking_time_limit == 'Less than 5mins':
        allName = allName[(allName['Total_Time']<=5)&(allName['Total_Time']>0)]
    elif cooking_time_limit == '5-15mins':
        allName = allName[(allName['Total_Time']>5)&(allName['Total_Time']<=15)]
    elif cooking_time_limit == '15-30mins':
        allName = allName[(allName['Total_Time']>15)&(allName['Total_Time']<=30)]
    elif cooking_time_limit == '30-60mins':
        allName = allName[(allName['Total_Time']>30)&(allName['Total_Time']<=60)]
    elif cooking_time_limit =='1-2hrs':
        allName = allName[(allName['Total_Time']>60)&(allName['Total_Time']<=120)]
    else:
        allName = allName[(allName['Total_Time']>120)&(allName['Total_Time']>0)]

    # check the word in name_lst, how many word is matching with other recipe
    allName['matching'] = allName['recipe_name_clean'].apply(lambda x: sum(
        [1 if lem.lemmatize(a) in name_lst else 0 for a in x.split(" ")])) 
    TopNRecommendation =  allName.sort_values(['matching','aver_rate'],ascending=False).head(5)
    TopNRecommendation_lst = TopNRecommendation.index.tolist()

    return TopNRecommendation_lst[:N]

def recommender_collabarative_by_recipeID(recipeID , N = 3):
    userRecommended = list(df_raw_rating[df_raw_rating['recipe_id']==recipeID]['user_id'])
    recipeRelated = list(df_raw_rating[df_raw_rating['user_id'].isin(userRecommended)]['recipe_id'])
    Good_Recipe_lst = list(df_raw_recipe[df_raw_recipe['aver_rate']>=4].index)
    df_Recommended = df_raw_rating[df_raw_rating['recipe_id'].isin(Good_Recipe_lst)]
    df_Recommended = df_raw_rating[df_raw_rating['recipe_id'].isin(recipeRelated)]
    df_Recommended = df_raw_rating[df_raw_rating['user_id'].isin(userRecommended)]

    #User-item-rating matrix
    userRecommendedMatrix = pd.DataFrame.pivot_table(
        df_Recommended,
        values='rating',
        index='user_id',
        columns='recipe_id',).fillna(0)

    recipeRecommended = list(userRecommendedMatrix.sum().sort_values(ascending=False).index)
    if recipeID in recipeRecommended:
        recipeRecommended.remove(recipeID)
    return recipeRecommended[:N]

#  --------------------------------- DOC2VEC FUNCTION ---------------------------------

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

def parseF(x):
    x = x.strip("[")
    x = x.strip("]")
    wordList = x.split(",")
    return wordList

@st.cache(persist=True,allow_output_mutation=True)
def get_data_df_all():
    url_all = "df_all_deploy2.csv"
    df_all=pd.read_csv(url_all, index_col="recipe_id")
    df_all["ingredients"] = df_all["ingredients"].apply(parseF)
    df_all["cooking_directions"] = df_all["cooking_directions"].apply(parseF)
    df_all["Vect_recipe_name"] = df_all["Vect_recipe_name"].apply(parseF)
    df_all["recipe_tag"] = df_all["recipe_tag"].apply(parseF)
    return df_all

df_all = get_data_df_all()

model_directions = Doc2Vec.load("doc2vec_model_final")

def docToVecAlgo(dataframe, recipe_id, n):
    test_doc = dataframe.loc[recipe_id]["cooking_directions"]
    originalData = dataframe[dataframe.index == recipe_id]
    test_doc_vector = model_directions.infer_vector(test_doc)
    predicted, proba = zip(*model_directions.docvecs.most_similar(positive=[test_doc_vector], topn=3*n))
    new_dict = dataframe.iloc[list(predicted)]
    new_dict = pd.concat([new_dict, originalData], axis=0)
    return new_dict

def recipe_name_recommender_123(dataframe, recipe_id,N=5):
    name_lst = dataframe.loc[recipe_id]['Vect_recipe_name']
    # create dataframe used to store the number of matching word
    allName = pd.DataFrame(df_all.index).set_index("recipe_id")
    allName['recipe_name'] = df_all[['recipe_name']]
    allName['Vect_recipe_name'] = df_all[['Vect_recipe_name']]

    # check the word in name_lst, how many word is matching with other recipe
    allName['matching'] = allName['Vect_recipe_name'].apply(lambda x: sum([1 if a in name_lst else 0 for a in x]) )
    # sort the allRecipes by distance and take N closes number of rows to put in the TopNRecommendation as the recommendations
    TopNRecommendation =  allName.sort_values(['matching'],ascending=False).head(N+1)
    TopNRecommendation =  TopNRecommendation[TopNRecommendation.index != recipe_id]

    return df_all.loc[TopNRecommendation.index]

def finalized(recipe_id, n=5):
    processing = docToVecAlgo(df_all,recipe_id,n*15)
    df_new = recipe_name_recommender_123(processing, recipe_id,n)
    return list(df_new.index)[:n]

# ------------------------------------ Side Bar  ------------------------------------
from PIL import Image
Logo_image = Image.open('NaNaLogo.PNG')
st.sidebar.image(Logo_image,width=200)

st.sidebar.subheader("Find a recipe...")
def_search_bar = ''
search_word = st.sidebar.text_input("", def_search_bar).lower()

#   Checkboxes to trigger feature
trigger_nutrition = st.sidebar.checkbox("Activate search by nutrition")
trigger_cooking_method = st.sidebar.checkbox("Search by Cooking Method")
trigger_cooking_time = st.sidebar.checkbox('Search by Cooking Time')

# ------------------------------------ Conditioning  ------------------------------------
if trigger_cooking_method ==False and trigger_nutrition==False and trigger_cooking_time==False and search_word==def_search_bar:
    st.write(':blush:   Hot Recipe of the month :blush:')
    recipe_list = recommender_hot_recipe()

elif trigger_cooking_method == False and trigger_nutrition==False and trigger_cooking_time==False and search_word!=def_search_bar:    
    recipe_list = recommender_recipe_name(search_word)
    st.write(recipe_list)

elif trigger_nutrition==True and trigger_cooking_method==False and trigger_cooking_time==False :
    #   Nutrition Input
    st.sidebar.title("Nutrition Ratio") 
    carbohydrates_input = st.sidebar.slider('carbohydrates:', min_value=0.0, max_value=1.0, step=0.1, key="1")
    fiber_input = st.sidebar.slider('fiber:', min_value=0.0, max_value=1.0, step=0.1, key="2")
    cholesterol_input = st.sidebar.slider('cholesterol:', min_value=0.0, max_value=1.0, step=0.1, key="3")
    protein_input = st.sidebar.slider('protein:', min_value=0.0, max_value=1.0, step=0.1, key="4")
    vitaminA_input = st.sidebar.slider('vitaminA:', min_value=0.0, max_value=1.0, step=0.1, key="5")
    vitaminB6_input = st.sidebar.slider('vitaminB6:', min_value=0.0, max_value=1.0, step=0.1, key="6")
    vitaminC_input = st.sidebar.slider('vitaminC:', min_value=0.0, max_value=1.0, step=0.1, key="7")
    calcium_input = st.sidebar.slider('calcium:', min_value=0.0, max_value=1.0, step=0.1, key="8")
    calories_input = st.sidebar.slider('calories:', min_value=0.0, max_value=1.0, step=0.1, key="9")
    fat_input = st.sidebar.slider('fat:', min_value=0.0, max_value=1.0, step=0.1, key="10")
    saturatedFat_input = st.sidebar.slider('saturatedFat:', min_value=0.0, max_value=1.0, step=0.1, key="11")
    caloriesFromFat_input = st.sidebar.slider('caloriesFromFat:', min_value=0.0, max_value=1.0, step=0.1, key="12")
    folate_input = st.sidebar.slider('folate:', min_value=0.0, max_value=1.0, step=0.1, key="13")
    sugars_input = st.sidebar.slider('sugars:', min_value=0.0, max_value=1.0, step=0.1, key="14")
    sodium_input = st.sidebar.slider('sodium:', min_value=0.0, max_value=1.0, step=0.1, key="15")
    thiamin_input = st.sidebar.slider('thiamin:', min_value=0.0, max_value=1.0, step=0.1, key="16")
    niacin_input = st.sidebar.slider('niacin:', min_value=0.0, max_value=1.0, step=0.1, key="17")
    magnesium_input = st.sidebar.slider('magnesium:', min_value=0.0, max_value=1.0, step=0.1, key="18")
    iron_input = st.sidebar.slider('iron:', min_value=0.0, max_value=1.0, step=0.1, key="19")
    potassium_input = st.sidebar.slider('potassium:', min_value=0.0, max_value=1.0, step=0.1, key="20")
    
    #   Modelling
    nutritions_lst = ['carbohydrates','fiber','cholesterol','protein','vitaminA','vitaminB6','vitaminC','calcium',\
        'calories','fat','saturatedFat','caloriesFromFat','folate','sugars','sodium','thiamin','niacin','magnesium','iron','potassium',]
    nutrition_input = {'niacin_input':niacin_input, 'sugars_input':sugars_input, 'sodium_input':sodium_input, 'carbohydrates_input':carbohydrates_input,\
        'vitaminB6_input':vitaminB6_input, 'calories_input':calories_input, 'thiamin_input':thiamin_input, 'fat_input':fat_input, 'folate_input':folate_input,
        'caloriesFromFat_input':caloriesFromFat_input, 'calcium_input':calcium_input, 'fiber_input':fiber_input, 'magnesium_input':magnesium_input,\
        'iron_input':iron_input, 'cholesterol_input':cholesterol_input, 'protein_input':protein_input, 'vitaminA_input':vitaminA_input, \
        'potassium_input':potassium_input, 'saturatedFat_input':saturatedFat_input, 'vitaminC_input':vitaminC_input }
    recipe_list = recommender_nutrition(nutrition_input,search_word)

    #   Display
    st.subheader("Top rated matchings:")

elif trigger_nutrition==False and trigger_cooking_method==True and trigger_cooking_time==False :
    st.sidebar.title("Pick a Cooking Method: ")
    cooking_options = ['baking','frying','roasting','grilling','steaming','poaching','simmering','broiling','stewing','braising']
    cooking_method_input = st.sidebar.selectbox(" ", cooking_options)
    recipe_list = recommender_CookingMethod(cooking_method_input,search_word)

elif trigger_nutrition==False and trigger_cooking_method==False and trigger_cooking_time==True :
    st.sidebar.title("Choose the Cooking Time ")
    cooking_time_input = st.sidebar.selectbox(" ", ['Less than 5mins','5-15mins','15-30mins','30-60mins','1-2hrs','More than 2hrs'])
    recipe_list = recommender_CookingTime(search_word, cooking_time_input)

else:
    st.write('Not supported multiple filters')
    recipe_list = recommender_hot_recipe()

#------------------------------------   Display Result   ------------------------------------

def show_result(recipe_id):
    # Part 1: Extract info from dataframe
    ingredients = df_raw_recipe.loc[df_raw_recipe.index == recipe_id]['ingredients'].squeeze()
    ingredients = ingredients.split("\n")

    cook_dir = df_raw_recipe.loc[df_raw_recipe.index == recipe_id]['cooking_directions'].squeeze()
    if cook_dir[0].isalpha() == False:  # ensure all unwanted characters are deleted
        cook_dir = cook_dir.strip(cook_dir[0])
    if cook_dir[len(cook_dir)-1] != ".":
        cook_dir = cook_dir.strip(cook_dir[len(cook_dir)-1])
    cook_dir = cook_dir.split("\\n")
    

    # Part 2: visualize the image and name of the recipe 
    st.markdown(df_raw_recipe[df_raw_recipe.index==recipe_id]['recipe_name'].values[0])
    st.image(get_image_url(recipe_id),width=128)   #, use_column_width=True
    st.write('Rating: {:.1f}'.format(df_raw_recipe[df_raw_recipe.index==recipe_id]['aver_rate'].values[0]))

    with st.beta_expander('Time Info', expanded=False):
        try:
            st.write('Prep Time: ', cook_dir[1])
        except:
            st.write('Prep Time: N/A')
        try:
            st.write(f"{cook_dir[2]} : {cook_dir[3]}")
        except:
            st.write('Cooking Time: N/A')

    with st.beta_expander('Ingredients', expanded=False):
        for i in ingredients:
            st.write(i)

    with st.beta_expander('Cooking method', expanded=False):
        for i in range(6, len(cook_dir)):
            st.write(cook_dir[i])

    with st.beta_expander('Nutrition', expanded=False):
        display_col_nutrition = ['carbohydrates','protein', 'vitaminA','vitaminB6', 'vitaminC', 'sugars', 'calories', 'fat', 'saturatedFat',\
       'folate', 'caloriesFromFat', 'calcium', 'fiber', 'magnesium', 'iron', 'cholesterol', 'potassium', 'niacin', 'sodium', 'thiamin']
        for _nutrition in display_col_nutrition:
            st.write(_nutrition, ': {:.2f} '.format(df_raw_recipe[df_raw_recipe.index == recipe_id][_nutrition].values[0]),\
                eval(df_raw_recipe[df_raw_recipe.index==recipe_id]['nutritions'].iloc[0])[_nutrition]['unit'])

    with st.beta_expander('Recommended by Others'):
        collab_recipe_list = recommender_collabarative_by_recipeID(recipe_id,3)
        for _count in range(0,len(collab_recipe_list)):
            st.image(get_image_url(collab_recipe_list[_count]),width=128, use_column_width=True)
            st.markdown(df_raw_recipe[df_raw_recipe.index==collab_recipe_list[_count]]['recipe_name'].values[0])

    with st.beta_expander('Related Recipe', expanded=False):
        closest_recipe_list = finalized(recipe_id,3)
        for _count in range(0,len(closest_recipe_list)):
            st.image(get_image_url(closest_recipe_list[_count]),width=128, use_column_width=True)
            st.markdown(df_raw_recipe[df_raw_recipe.index==closest_recipe_list[_count]]['recipe_name'].values[0])

for x in range(0,len(recipe_list)):
    show_result(recipe_list[x])




#------------------------------------   Remark   ------------------------------------

def to_improve():
    '''
    Undone:
    DOC2VEC

    Add:
    Hybrid ( library should be spice?)

    Improve:
    1.  Clean and add comment record in the recipe search result
    2.  The more words in a recipe name, the lesser distance they have during name search
    3.  Fix cooking time (Data clean was not audited)
    4.  CSS styling

    '''

def to_merge():
    

    '''
    #   --------------------------------------------------to merge----------------------------------------------------------------

    # GET INGRIDENT & COOKING DIRECTION DETAIL
    #@st.cache(persist=True,allow_output_mutation=True)
    #def get_recipe():
    #    return pd.read_csv("raw_recipe_clean.csv")
    #recipe_df = get_recipe()

    st.dataframe(df_raw_recipe.head(5))
    st.write(df_raw_recipe.loc[222388]['ingredients_to_show'])
    st.write(df_raw_recipe.loc[222388]['cooking_directions_to_show'])







    from gensim.test.utils import common_texts
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.test.utils import get_tmpfile

    #df_ = pd.read_csv("./core-data_recipe.csv", index_col="recipe_id")

    def parseF(x):
        x = x.strip("[")
        x = x.strip("]")
        wordList = x.split(",")
        return wordList
    @st.cache(persist=True,allow_output_mutation=True)
    def get_data_df_all():
        url_all = "df_all_deploy2.csv"             # "./David/df_all.csv"
        df_all=pd.read_csv(url_all, index_col="recipe_id")
        df_all["ingredients"] = df_all["ingredients"].apply(parseF)
        df_all["cooking_directions"] = df_all["cooking_directions"].apply(parseF)
        df_all["Vect_recipe_name"] = df_all["Vect_recipe_name"].apply(parseF)
        df_all["recipe_tag"] = df_all["recipe_tag"].apply(parseF)
        return df_all
    df_all = get_data_df_all()

    #st.dataframe(df_all.head(5))

    model_directions = Doc2Vec.load("David/doc2vec_model_final")

    def docToVecAlgo(dataframe, recipe_id, n):
        test_doc = dataframe.loc[recipe_id]["cooking_directions"]
        originalData = dataframe[dataframe.index == recipe_id]
        test_doc_vector = model_directions.infer_vector(test_doc)
        predicted, proba = zip(*model_directions.docvecs.most_similar(positive=[test_doc_vector], topn=3*n))
        new_dict = dataframe.iloc[list(predicted)]
        new_dict = pd.concat([new_dict, originalData], axis=0)
        return new_dict

    def recommender_recipe_name_123(dataframe, recipe_id,N=5):
        name_lst = dataframe.loc[recipe_id]['Vect_recipe_name']
        allName = pd.DataFrame(df_all.index).set_index("recipe_id")
        allName['recipe_name'] = df_all[['recipe_name']]
        allName['Vect_recipe_name'] = df_all[['Vect_recipe_name']]
        allName['matching'] = allName['Vect_recipe_name'].apply(lambda x: sum([1 if a in name_lst else 0 for a in x]) )
        TopNRecommendation =  allName.sort_values(['matching'],ascending=False).head(N+1)
        TopNRecommendation =  TopNRecommendation[TopNRecommendation.index != recipe_id]
        return df_all.loc[TopNRecommendation.index]

    def finalized(recipe_id, n=5):
        processing = docToVecAlgo(df_all,recipe_id,n*15)
        df_new = recommender_recipe_name_123(processing, recipe_id,n)
        return list(df_new.index)[:n]

    x = 78299
    closest_recipe_list = finalized(x,5)

    st.write("Check:    ")
    st.write(df_all[df_all.index==x]['recipe_name'])
    st.write(closest_recipe_list)
    st.write('df_all shape' , df_all.shape)

    def load_recipe(recipe_id):
        recipe_df = df_raw_recipe
        st.title(recipe_id)

        # Select info from dataframe
        ingredients = recipe_df.loc[recipe_df.index==recipe_id]['ingredients'].squeeze()
        cook_dir = recipe_df.loc[recipe_df.index==recipe_id]['cooking_directions'].squeeze()

        # Expander
        with st.beta_expander('Ingredients'):
            st.write(ingredients)
        with st.beta_expander('Cooking Direction'):
            st.write(cook_dir)
        finalized(recipe_id)
        return

    st.write("check 78299:")
    load_recipe(78299)

    '''
