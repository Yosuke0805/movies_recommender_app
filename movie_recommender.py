''' coding: utf-8 '''
# ------------------------------------------------------------
# Content : Creating a tool for recommending movies by contents based
# Author : Yosuke Kawazoe
# Data Updated：18/09/2024
# Update Details：
# ------------------------------------------------------------

# Import
import os
import streamlit as st
import traceback
import tempfile
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# Used to securely store your API key
import google.generativeai as genai

# Config
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# ★★★★★★  main part ★★★★★★
# ------------------------------------------------------------
def main():

    try:
        # Set the title and description
        st.title("🎥 Recommending movies")

        # set Gemini API
        # GOOGLE_API_KEY = st.sidebar.text_input("Input your Google AI Studio API", type="password")
        # genai.configure(api_key=GOOGLE_API_KEY)

        # Input data
        gen_md_df = pd.read_csv('data/genre_mv.csv')
        indice_df = pd.read_csv('data/indices.csv')
        smd_meta_df = pd.read_csv('data/smd_meta.csv')
        indice = pd.Series(indice_df.reset_index().index, index=indice_df['title'])
        # cosine_sin_combined = np.load('data/cosine_sim_combined.npy', allow_pickle=False)
        # cosine_sin_combined_df = pd.read_parquet('data/cosine_sim_combined.parquet')
        # cosine_sin_combined = cosine_sin_combined_df.to_numpy()
        tokenizer_df = pd.read_parquet('data/tokenized_by_bert.parquet')
        cosine_sin_combined = get_emsemble_cosine_matrix(tokenizer_df, smd_meta_df)


        genre_list = gen_md_df.genre.unique().tolist()
        chosen_genre = st.selectbox(label="Select genre", options=genre_list)

        if chosen_genre:
            genre_top_df = build_chart(gen_md_df, chosen_genre)
            the_number_of_movies = st.number_input(label="The number of movies options", min_value=1, max_value=50, value=10)
            top_movies_per_genre_list = genre_top_df['title'][:the_number_of_movies].tolist()
            favorite_movie = st.selectbox(label="Select your favorite movie", options=top_movies_per_genre_list)
            if favorite_movie:
                with st.form("summary_form", clear_on_submit=False):
                    submitted = st.form_submit_button('Show movies recommendations')
                    if submitted:
                        recommendation_df = improved_recommendations(indice, cosine_sin_combined, favorite_movie, smd_meta_df)
                        recommendation_df = recommendation_df['title'].reset_index(drop=True).reset_index().rename(columns={'index': 'rank'})
                        st.dataframe(recommendation_df)
            else:
                st.error("Please select your favorite movie!!", icon="🚨")
        else:
            st.error("Please choose your genre!!", icon="🚨")
    except Exception as e:
        st.error("An unexpected error occurred in the main function.", icon="🚨")
        st.error(f"Details: {str(e)}")
        logger.error(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()


# ------------------------------------------------------------
# ★★★★★★  functions ★★★★★★
# ------------------------------------------------------------
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_recommendations_with_weighted_rating(df, top_n=250, percentile=0.95):
    columns = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']
    C = df.vote_average.mean()
    print('Average vote:',C)
    m = df.vote_count.quantile(percentile)
    print(f'{percentile*100}th percentile of vote count:',m)
    qualified_df = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][columns]
    
    qualified_df['wr'] = qualified_df.apply(weighted_rating, axis=1, m=m, C=C)
    qualified_df = qualified_df.sort_values('wr', ascending=False).head(top_n)
    
    return qualified_df

def build_chart(df, genre, percentile=0.85):
    try:
        df = df[df['genre'] == genre]
        df = df.rename({'genre':'genres'},axis=1)
        qualified_df = get_recommendations_with_weighted_rating(df, top_n=250, percentile=percentile)
    
        return qualified_df
    except Exception as e:
        logger.error(f"Error in improved_recommendations: {str(e)}")
        raise

def get_emsemble_cosine_matrix(tokenized_df, smd_meta):
    # calculate cosine similarity with moview overview data vectozied by BERT
    cosine_sim_bert = cosine_similarity(tokenized_df, tokenized_df)
    
    # calculate cosine similarity with director data vectozied by CountVectorizer
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1, stop_words='english')
    count_matrix = count.fit_transform(smd_meta['soup'])
    cosine_sim_director = cosine_similarity(count_matrix, count_matrix)

    # combine the two similarity matrices which captures overview similarity and director similarity
    cosine_sim_combined = 0.5 * cosine_sim_bert + 0.5 * cosine_sim_director
    return cosine_sim_combined

def improved_recommendations(indices, cosine_df, title, smd_meta):
    try:
        # get index of the movie
        idx = indices[title]
        # get similar movies
        sim_scores = list(enumerate(cosine_df[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # extract top 30
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        top30_similar_movies_df = smd_meta.iloc[movie_indices]
        qualified_df = get_recommendations_with_weighted_rating(top30_similar_movies_df, top_n=30, percentile=0.60)

        return qualified_df
    except Exception as e:
        logger.error(f"Error in improved_recommendations: {str(e)}")
        raise


# ------------------------------------------------------------
# ★★★★★★  execution part  ★★★★★★
# ------------------------------------------------------------
if __name__ == '__main__':

    # execute
    main()
