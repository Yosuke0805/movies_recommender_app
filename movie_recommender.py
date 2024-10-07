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

# Config
genre_mv_path = 'data/genre_mv.csv'
indice_path = 'data/indices.csv'
smd_meta_path = 'data/smd_meta.csv'
tokenizer_path = 'data/tokenized_by_bert.parquet'

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

        # Input data
        gen_md_df = load_data(genre_mv_path)
        indice_df =load_data(indice_path)
        movie_director_actor_keywords_df = load_data(smd_meta_path)
        tokenized_by_bert_df = load_data(tokenizer_path)

        # transfrom data
        movie_title_indices = pd.Series(indice_df.reset_index().index, index=indice_df['title'])
        cosine_sim_combined_array = get_emsemble_cosine_matrix(tokenized_by_bert_df, movie_director_actor_keywords_df)

        genre_list = gen_md_df['genre'].unique().tolist()
        # let users select genre on select box which starts from Action genre
        chosen_genre = st.selectbox(label="Select genre", options=genre_list, index=7)

        if chosen_genre:
            # select movies based on genre
            genre_top_df = select_top_movies_by_genre(gen_md_df, chosen_genre)
            the_number_of_movies = st.number_input(label="The number of options for movies", min_value=1, max_value=50, value=20)
            top_movies_per_genre_list = genre_top_df['title'][:the_number_of_movies].tolist()
            user_favorite_movie = st.selectbox(label="Select your favorite movie", options=top_movies_per_genre_list)
            if user_favorite_movie:
                with st.form("summary_form", clear_on_submit=False):
                    submitted = st.form_submit_button('Show movies recommendations')
                    if submitted:
                        recommendation_df = improved_recommendations(movie_title_indices, cosine_sim_combined_array, user_favorite_movie, movie_director_actor_keywords_df)
                        recommendation_df = recommendation_df['title'].reset_index(drop=True).reset_index().rename(columns={'index': 'rank'})
                        recommendation_df['rank'] +=1
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
@st.cache_data
def load_data(file_path):
    """
    load data by using cashe.

    Args:
        file_path (str): The file path.

    Returns:
        dataframe: The dataframe.
    """
    if file_path.endswith('.csv'):
        # Load CSV file
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        # Load Parquet file
        df = pd.read_parquet(file_path)
    return df

def weighted_rating(x, m, C):
    """
    helper function for extract_qualified_movies_sorted_by_weighted_rating to calculate weighted rating.

    Args:
        x (dataframe): dataframe.
        m (float): The percentile of vote counts.
        C (float): The mean of vote average.

    Returns:
        wr(series): The series.
    """
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def extract_qualified_movies_sorted_by_weighted_rating(df, top_n=100, percentile=0.85):
    """
    helper function for select_top_movies_by_genre and improved_recommendations to extract qualified movies by weighted rating.

    Args:
        df (dataframe): dataframe.
        top_n (int): The number of movies to show.
        percentile (float): The percentile for vote counts.

    Returns:
        qualified_df(dataframe): The dataframe which was extracted by weighted rating.
    """
    columns = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']
    C = df.vote_average.mean()
    # print('Average vote:',C)
    m = df.vote_count.quantile(percentile)
    # print(f'{percentile*100}th percentile of vote count:',m)
    qualified_df = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][columns]
    
    qualified_df['wr'] = qualified_df.apply(weighted_rating, axis=1, m=m, C=C)
    qualified_df = qualified_df.sort_values('wr', ascending=False).head(top_n)
    
    return qualified_df

@st.cache_data
def select_top_movies_by_genre(df, genre):
    try:
        df = df[df['genre'] == genre]
        df = df.rename({'genre':'genres'},axis=1)
        qualified_df = extract_qualified_movies_sorted_by_weighted_rating(df)
    
        return qualified_df
    except Exception as e:
        logger.error(f"Error in select_top_movies_by_genre: {str(e)}")
        raise

@st.cache_data
def get_emsemble_cosine_matrix(tokenized_by_bert_df, movie_director_actor_keywords_df):
    # calculate cosine similarity with moview overview data vectozied by BERT
    cosine_sim_bert = cosine_similarity(tokenized_by_bert_df, tokenized_by_bert_df)
    
    # calculate cosine similarity with director data vectozied by CountVectorizer
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1, stop_words='english')
    count_matrix = count.fit_transform(movie_director_actor_keywords_df['soup'])
    cosine_sim_director = cosine_similarity(count_matrix, count_matrix)

    # combine the two similarity matrices which captures overview similarity and director similarity
    cosine_sim_combined = 0.5 * cosine_sim_bert + 0.5 * cosine_sim_director
    return cosine_sim_combined

def improved_recommendations(indices, cosine_combined_array, movie_title, movie_director_actor_keywords_df):
    try:
        # get index of the movie
        idx = indices[movie_title]
        # get similar movies
        sim_scores = list(enumerate(cosine_combined_array[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # extract top 30
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        top30_similar_movies_df = movie_director_actor_keywords_df.iloc[movie_indices]
        qualified_df = extract_qualified_movies_sorted_by_weighted_rating(top30_similar_movies_df, top_n=30, percentile=0.60)

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
