import os
import sys
sys.path.append(os.path.dirname(__file__))
import pandas as pd
# from data_loader import DataLoader
# from process_data import ProcessData
import joblib
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import json
import requests
from openai import OpenAI
from coaches import Coaches
from model_trainer import ModelTrainer
from run import RunPipeline
from added_features import (
distance_bin, field_position, score_difference, fourth_and_one, compute_distance_success_rates,
score_diff_subtraction, add_yardline_100, get_knn_features,
drive_inside20, game_half, half_seconds_remaining, game_seconds_remaining)
from config import MODEL_SIMULATION_COLUMNS
from utils import run_streamlit_preloads, model_misses, scenario_sentence
from io import BytesIO

st.set_page_config(page_title="NFL 4th Down Analytics Model", layout="wide")
st.title("NFL 4th Down Analytics Model")

client = OpenAI(api_key=st.secrets["api"]["OPENAI_API_KEY"])

# @st.cache_data
# def get_data_loader():
#     from data_loader import DataLoader
#     return DataLoader()

# data_loader = get_data_loader()

# dataset_url = "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/dataset.parquet"
#
# @st.cache_data
# def load_data():
#     """Load preprocessed data from GitHub Releases."""
#     try:
#         response_dataset = requests.get(dataset_url)
#         df = pd.read_parquet(BytesIO(response_dataset.content), engine='pyarrow')
#         return df
#     except Exception as e:
#         st.error(f"Failed to load data: {e}")
#         return pd.DataFrame()
#
# df = load_data()
# if df.empty:
#     st.warning("No data available to display.")


# pipeline.run_pipeline()

# @st.cache_data
# def get_process_data():
#     from process_data import ProcessData
#     return ProcessData()

pipeline_urls = {
    "df": "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/pipeline_df.pkl",
    "df_model": "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/pipeline_df_model.pkl",
    "df_punt_fg": "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/pipeline_df_punt_fg.pkl",
    "base_pred_df": "https://github.com/ddamanze/NFL_Fourth_Down_Decision_Model/releases/download/v1.0-datasets/pipeline_base_pred_df.pkl"
}

@st.cache_data
def load_pipeline_outputs():
    outputs = {}
    for name, url in pipeline_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()  # crash if download fails
            outputs[name] = joblib.load(BytesIO(response.content))
        except Exception as e:
            st.error(f"Failed to load {name}: {e}")
            outputs[name] = None

    # Ensure expected columns exist in df
    if outputs["df"] is not None:
        for col in ['model_recommendation', 'decision_class', 'year']:
            if col not in outputs["df"].columns:
                outputs["df"][col] = None

    return outputs["df"], outputs["df_model"], outputs["df_punt_fg"], outputs["base_pred_df"]

df, df_model, df_punt_fg, base_pred_df = load_pipeline_outputs()

pipeline = RunPipeline(df_model, mode='realtime')

# Inject CSS once at the top of your app
st.markdown("""
    <style>
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

tabs = st.tabs([
    "Home Page",
    "Coach Aggression Assessment",
    "4th Down Model Predictor",
    "Scouting Report"
])
tab1, tab2, tab3, tab4 = tabs
current_tab_index = st.session_state.active_tab

with tabs[current_tab_index]:
    pass

with tab1:
    st.markdown(
        """
        <div style="text-align: center; margin: auto; padding: 20px;">
            Should they go for it here? A question that every NFL fan has asked throughout the course of a game 
            during football season. There have been many controversial coaching decisions on 4th down that have 
            impacted the outcome of a game. While some coaches still rely on their gut instincts, other coaches have 
            shifted towards an analytical approach. With machine learning and analytics, I have created a model that 
            provides a recommendation in any given scenario during a game. With this model I assessed which coaches 
            tend to be more aggressive in 4th down situations, and who tends to lean more conservatively.
        </div>
        """,
        unsafe_allow_html=True
    )

    query_to_key_plays = {
        # "Colts Stop the Patriots on 4th & 2": [
        #     {"label": "🚀 4th & 2 Failure", "timestamp": 1, "description": "A decision many people believe "
        #                                                                   "sparked the start of the 4th down decision "
        #                                                                   "debates. Tom Brady vs Peyton Manning. Coach Belicheck "
        #                                                                   "decides to roll the dice and go for it. Manning and the "
        #                                                                   "Colts go on to win the game."}
        # ],
        "Buccaneers vs. Packers NFC Championship Game Highlights | NFL 2020 Playoffs": [
            {"label": "🚀 Kicked FG", "timestamp": 645,
             "description": "Tom Brady vs Aaron Rodgers. 2 minutes left, down by 8. "
                            "Coach McCarthy decided to kick a FG and the Packers "
                            "never get the ball back as Tom Brady and the "
                            "Buccaneers run the clock out.",
             "play_id": 3728, "game_id": "2020_20_TB_GB", "coach_decision": "Kick FG"}
        ],
        "Detroit Lions vs. San Fransisco 49ers NFC Championship FULL GAME | 2023 NFL Postseason": [
            {"label": "🚀 4th & 2 Failure", "timestamp": 4565,
             "description": "The Lions were up 14 in the 3rd quarter with the opportunity to make it a 3 score game. "
                            "Coach Campbell decided to go for it from the 28 yard line instead of kicking. "
                            "Some argue that this potentially shifted the momentum of the game to the Niners with a 4th down stop. "
                            "What did analytics say about this decision?",
             "play_id": 2682, "game_id": "2023_21_DET_SF", "coach_decision": "Go For It"},
            {"label": "🚀 4th & 3 Failure", "timestamp": 6271,
             "description": "The Niners have gained momentum and taken the lead. "
                            "Detroit has the ball, and it's 4th and 3 from the Niners' 30 yard line. 7 minutes left, down by 3. "
                            "Coach Campbell decided to go for it once again instead of kicking for the tie. "
                            "Was this the right call? ",
             "play_id": 3608, "game_id": "2023_21_DET_SF", "coach_decision": "Go For It"}

        ],
        "Buffalo Bills vs. Kansas City Chiefs FULL GAME | AFC Championship NFL 2024 Season": [
            {"label": "🚀 4th & 1 Failure", "timestamp": 5035, "description": "Many debate whether or not the Bills "
                                                                             "got the 1st down, but was it the right "
                                                                             "decision to go for it?",
             "play_id": 3334, "game_id": "2024_21_BUF_KC", "coach_decision": "Go For It"}
        ]
    }


    def get_youtube_videos(api_key, query, key_plays, max_results=1):
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        videos = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            thumbnail = item["snippet"]["thumbnails"]["medium"]["url"]

            videos.append({
                "title": title,
                "thumbnail": thumbnail,
                "video_id": video_id,
                "key_plays": key_plays
            })
        return videos


    # Example usage
    api_key = st.secrets["api"]["google_api_key"]
    queries = [  # "Colts Stop the Patriots on 4th & 2",
        "Buccaneers vs. Packers NFC Championship Game Highlights | NFL 2020 Playoffs",
        "Detroit Lions vs. San Fransisco 49ers NFC Championship FULL GAME | 2023 NFL Postseason",
        "Buffalo Bills vs. Kansas City Chiefs FULL GAME | AFC Championship NFL 2024 Season"
    ]
    for query in queries:
        key_plays = query_to_key_plays.get(query, [])  # fallback to empty if none
        videos = get_youtube_videos(api_key, query, key_plays)
        # Get game and play id to pull decisions for all of these
        for video in videos:
            with st.container():
                # Center image + text
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 15px 0;">
                        <img src="{video['thumbnail']}" width="400"><br>
                        <strong>{video['title']}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Center key plays
                for play in video["key_plays"]:
                    url = f"https://www.youtube.com/watch?v={video['video_id']}&t={play['timestamp']}"
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 10px 10px;">
                            {play['description']}<br>
                            - <a href="{url}" target="_blank">{play['label']}</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # --- Reveal button ---

                    if st.button("Reveal Model Recommendation", key=f"{video['title']}_{play['label']}"):
                        rec_row = base_pred_df[
                            (base_pred_df["play_id"] == play["play_id"]) &
                            (base_pred_df["game_id"] == play["game_id"])
                            ]
                        if not rec_row.empty:
                            model_recommendation = rec_row.iloc[0]["model_recommendation"]
                        else:
                            model_recommendation = "No recommendation available."

                        # Determine color based on alignment
                        if "No recommendation" in model_recommendation:
                            color_bg = "#f8f9fa"  # neutral light gray
                            color_text = "#6c757d"
                        elif model_recommendation.lower() == play.get("coach_decision", "").lower():
                            color_bg = "#d4edda"  # light green
                            color_text = "#155724"
                        else:
                            color_bg = "#f8d7da"  # light red
                            color_text = "#721c24"

                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: center;">
                                <div style="
                                    text-align: center;
                                    background-color: {color_bg};
                                    color: {color_text};
                                    font-weight: bold;
                                    font-size: 18px;
                                    padding: 15px;
                                    border-radius: 10px;
                                    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
                                    margin: 10px auto;
                                    display: inline-block;
                                    max-width: 600px;
                                ">
                                    {model_recommendation}
                                </div>
                            </div>

                            """,
                            unsafe_allow_html=True
                        )

with tab2:
    selected_year_sidebar = st.selectbox("Select a season", [sorted(df['year'].unique(), reverse=True)[0]])#.tolist())#index = len(years)-1)
    coaches = Coaches(post_pred_df=base_pred_df, latest_season=selected_year_sidebar)

    coach_stats_df = coaches.coaching_stats()

    net_aggressive_df = coach_stats_df[['posteam_coach', 'net_aggressiveness_score', 'alignment_rate']].rename(
        columns={'posteam_coach': 'coach'})
    coach_model_align_df = coach_stats_df[['posteam_coach', 'alignment_rate']].rename(
        columns={'posteam_coach': 'coach'})

    # Load the mapping
    with open('coach_images.json', 'r') as f:
        image_map = json.load(f)
    coaches_df = pd.DataFrame(image_map)

    default_img = "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"
    net_aggressive_df['team'] = net_aggressive_df['coach'].map(coaches_df.set_index('coach')['team'])
    net_aggressive_df['coach_url'] = net_aggressive_df['coach'].map(coaches_df.set_index('coach')['url'])

    with open('team_logos.json', 'r') as f:
        team_map = json.load(f)
    net_aggressive_df['team_logo_url'] = net_aggressive_df['team'].map(team_map)

    net_aggressive_df['net_aggressive_score'] = net_aggressive_df['net_aggressiveness_score'].round(3)
    top5 = net_aggressive_df.sort_values(by='net_aggressive_score', ascending=False).head(5)
    bottom5 = net_aggressive_df.sort_values(by='net_aggressive_score', ascending=True).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Aggressive Coaches")
        for _, row in top5.iterrows():
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; padding: 6px 0;">
                    <img src="{row['coach_url']}" style="width:100px; border-radius:8px; margin-bottom: 8px; margin-right: 12px;" />
                    <div style="font-weight: bold; font-size: 16px; margin-right: 12px;">{row['coach']}</div>
                    <div style="color: #ccc; font-size: 12px; margin-bottom: 4px; margin-right: 2px;">{row['team']}</div>
                    <img src="{row['team_logo_url']}" style="width:32px; margin-top: 2px;" />
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.subheader("Most Conservative Coaches")
        for _, row in bottom5.iterrows():
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; padding: 6px 0;">
                    <img src="{row['coach_url']}" style="width:100px; border-radius:8px; margin-bottom: 8px; margin-right: 12px;" />
                    <div style="font-weight: bold; font-size: 16px; margin-right: 12px;">{row['coach']}</div>
                    <div style="color: #ccc; font-size: 12px; margin-bottom: 4px; margin-right: 2px;">{row['team']}</div>
                    <img src="{row['team_logo_url']}" style="width:32px; margin-top: 2px;" />
                </div>
                """,
                unsafe_allow_html=True
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=net_aggressive_df['alignment_rate'],
        y=net_aggressive_df['net_aggressiveness_score'],
        mode='markers',
        marker=dict(size=20, color='rgba(0,0,0,0)'),  # invisible points
        hovertext=net_aggressive_df['coach'],
        hoverinfo='text'
    ))

    fig.update_layout(
        images=[
            dict(
                source=logo,
                x=x, y=y,
                xref="x", yref="y",
                sizex=0.035, sizey=0.035,
                xanchor="center",
                yanchor="middle",
                layer="above"
            ) for x, y, logo in zip(net_aggressive_df['alignment_rate'], net_aggressive_df['net_aggressiveness_score'], net_aggressive_df['team_logo_url'])
        ],
        xaxis_title="Alignment Rate",
        yaxis_title="Aggressive Score",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )

    fig.update_xaxes(range=[net_aggressive_df['alignment_rate'].min()-0.01, net_aggressive_df['alignment_rate'].max()+0.01])
    fig.update_yaxes(range=[net_aggressive_df['net_aggressiveness_score'].min()-0.05, net_aggressive_df['net_aggressiveness_score'].max()+0.05])

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='margin-bottom: 20px;'>"
                "ℹ️ This chart compares coach's aggressiveness and how often their decisions they align with the model. Coaches in the top right are aggressive in the right situations."
                "</div>", unsafe_allow_html=True)

    st.subheader("Coaching Decision Metrics")
    st.dataframe(coach_stats_df)

with tab3:
    st.markdown("""
        <style>
        .help-icon {
            display: inline-block;
            cursor: pointer;
            border-radius: 50%;
            background: #2a2a2a;
            color: #aaa;
            padding: 0 4px;
            font-size: 12px;
            margin-left: 8px;
        }
        .help-icon:hover::after {
            content: attr(data-tip);
            position: absolute;
            background: #333;
            color: white;
            padding: 5px 8px;
            border-radius: 5px;
            margin-top: -35px;
            margin-left: 10px;
            white-space: nowrap;
            z-index: 1000;
        }
        </style>

        <h3>
            Pre-Loaded Scenarios
            <span class="help-icon" data-tip="Run a randomized scenario pulled from the dataset. Teams and situations may vary.">?</span>
        </h3>
    """, unsafe_allow_html=True)

    # Pre-loaded scenarios. Takes a random row based on whatever scenario chosen by the user (quarter, score will vary)
    pre_loaded_score_diff = st.selectbox("Score Differential",['Trailing', 'Tie Game', 'Leading'])
    pre_loaded_scenario = st.selectbox("Select 4th down scenario", ['4th & 1 OWN 40YD Line',
                                              '4th & short 50YD Line',
                                              '4th & 1 OPP 40YD Line',
                                              '4th & short in FG Range',
                                              '4th & 1 in FG Range',
                                              '4th & Goal 3YD Line'])

    # Only regenerate pre_loaded_df if inputs change
    pre_loaded_df = run_streamlit_preloads(df_model, pre_loaded_scenario, pre_loaded_score_diff)
    if st.button("Run Model"):
        st.session_state.active_tab = 2
        pre_loaded_df = pipeline.run_pipeline(pre_loaded_df)
        pre_loaded_recommendation = pre_loaded_df['model_recommendation'].iloc[0]
        fourth_down_probability = pre_loaded_df['fourth_down_probability'].iloc[0]
        successful_wp = pre_loaded_df['fourth_success'].iloc[0]
        failure_wp = pre_loaded_df['fourth_failure'].iloc[0]
        fg_prob = pre_loaded_df['fg_prob'].iloc[0]

        if pre_loaded_recommendation == 'Go For It':
            st.write(
                f"Model Recommendation: {pre_loaded_recommendation}, "
                f"with a {round(fourth_down_probability * 100, 2)}% chance of converting ✅"
            )
            st.write(f"Win Probability if successful: {round(successful_wp * 100, 2)}%")
            st.write(f"Win Probability if failed: {round(failure_wp * 100, 2)}%")
        elif pre_loaded_recommendation == 'Kick FG':
            st.write(
                f"Model Recommendation: {pre_loaded_recommendation}, "
                f"with a {round(fg_prob * 100, 2)}% chance of making FG attempt✅"
            )


        # if pre_loaded_recommendation == 'Go For It':
        #     st.write(f"Model Recommendation: {pre_loaded_recommendation}, with a {round(fourth_down_probability * 100, 2)}% chance of converting ✅")
        #     st.write(f"Win Probability if successful: {round(successful_wp * 100, 2)}%")
        #     st.write(f"Win Probability if failed: {round(failure_wp * 100, 2)}%")
        # elif pre_loaded_recommendation == 'Kick FG':
        #     st.write(f"Model Recommendation: {pre_loaded_recommendation}, with a {round(fg_prob * 100, 2)}% chance of making FG attempt✅")

    with st.container():
        st.markdown("""
            <style>
            .help-icon {
                display: inline-block;
                cursor: pointer;
                border-radius: 50%;
                background: #2a2a2a;
                color: #aaa;
                padding: 0 4px;
                font-size: 12px;
                margin-left: 8px;
            }
            .help-icon:hover::after {
                content: attr(data-tip);
                position: absolute;
                background: #333;
                color: white;
                padding: 5px 8px;
                border-radius: 5px;
                margin-top: -35px;
                margin-left: 10px;
                white-space: nowrap;
                z-index: 1000;
            }
            </style>

            <h3>
                Batch Predictions
                <span class="help-icon" data-tip="Download the template and upload a file of scenarios to see model recommendations.">?</span>
            </h3>
        """, unsafe_allow_html=True)
        with open('4th_down_input_template.csv', 'r') as f:
            batch_predictions_template = f.read()
        st.download_button("Batch Predictions Template", data = batch_predictions_template,
                           file_name="4th_down_input_template.csv", mime="text/csv")
        uploaded_file = st.file_uploader("Batch Predictions Available; Upload a CSV File")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            batch_df = batch_df.rename(columns={'distance':'ydstogo'})
            #add everything from form version below
            batch_df = (batch_df.pipe(score_diff_subtraction).pipe(distance_bin).pipe(field_position).pipe(score_difference)
                        .pipe(fourth_and_one).pipe(drive_inside20)
                        .pipe(game_half).pipe(half_seconds_remaining).pipe(game_seconds_remaining).pipe(add_yardline_100))
            distance_success_dict = df_model.set_index('distance_bin')['distance_success_rate'].to_dict()
            batch_df['distance_success_rate'] = batch_df['distance_bin'].map(distance_success_dict)
            """Still need all knn entered info"""
            knn_batch_df = get_knn_features(df_model, batch_df)
            batch_df = pd.concat([batch_df, knn_batch_df], axis=1)
            batch_df = batch_df[MODEL_SIMULATION_COLUMNS]
            batch_df = pipeline.run_pipeline(batch_df)
            st.download_button("Download Predictions", batch_df.to_csv("4th_down_batch_predictions.csv", index=False))

    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
                    <style>
                    .help-icon {
                        display: inline-block;
                        cursor: pointer;
                        border-radius: 50%;
                        background: #2a2a2a;
                        color: #aaa;
                        padding: 0 4px;
                        font-size: 12px;
                        margin-left: 8px;
                    }
                    .help-icon:hover::after {
                        content: attr(data-tip);
                        position: absolute;
                        background: #333;
                        color: white;
                        padding: 5px 8px;
                        border-radius: 5px;
                        margin-top: -35px;
                        margin-left: 10px;
                        white-space: nowrap;
                        z-index: 1000;
                    }
                    </style>

                    <h3>
                        Enter Custom Scenario
                        <span class="help-icon" data-tip="Enter a custom scenario to see what the model would recommend.">?</span>
                    </h3>
                """, unsafe_allow_html=True)
        with st.form("Manually Enter a 4th Down Scenario",enter_to_submit=False):
            down = st.selectbox("Down",[4])
            distance = st.number_input("Distance to go", min_value=1, max_value=99, value=10)
            field_position = st.selectbox("Field Position", ["Own", "Opp", "Midfield"])
            field_position_yards = st.number_input("Field position yardage", min_value=1, max_value=50, value=50)
            pos_team_score = st.number_input("Possession Team's Score", value = 0)
            def_team_score = st.number_input("DEF Score", value = 0)
            minutes = st.number_input("Minutes left", min_value=0, max_value=15, value=15)
            seconds = st.number_input("Seconds left", min_value=0, max_value=59, value=0)
            qtr = st.selectbox("QTR",[1,2,3,4])
            overtime = st.selectbox("Overtime?", ["No", "Yes"])
            home_or_away = st.selectbox("Home or Away", ["Home", "Away"])
            # If OT tell them to put 4th

            submitted = st.form_submit_button(label="Run Model")

        if submitted:
            st.session_state.active_tab = 2
            manual_input_df = pd.DataFrame({
                "down": int(down),
                "ydstogo": int(distance),
                "yardline_100": int(field_position_yards),
                "game_half": game_half,
                "qtr": int(qtr),
                "posteam_type": home_or_away,
                "posteam_score": int(pos_team_score),
                "defteam_score": int(def_team_score),
                "drive_inside20": drive_inside20,
                "field_position":  field_position,
                "overtime": overtime,
                "minutes": int(minutes),
                "seconds": int(seconds)
            }, index=[0])

            manual_input_df = (manual_input_df.pipe(score_diff_subtraction).pipe(score_difference)
                        .pipe(fourth_and_one).pipe(add_yardline_100).pipe(drive_inside20)
                        .pipe(game_half).pipe(half_seconds_remaining).pipe(game_seconds_remaining).pipe(distance_bin))
            distance_success_dict = df_model.set_index('distance_bin')['distance_success_rate'].to_dict()
            manual_input_df['distance_success_rate'] = manual_input_df['distance_bin'].map(distance_success_dict).astype(float)

            knn_manual_df = get_knn_features(df_model, manual_input_df)
            manual_input_df = pd.concat([manual_input_df, knn_manual_df], axis=1)
            manual_input_df = manual_input_df[MODEL_SIMULATION_COLUMNS]
            input_pred = pipeline.run_pipeline(manual_input_df)
            recommendation_str = input_pred['model_recommendation'].iloc[0]
            fourth_down_probability = input_pred['fourth_down_probability'].iloc[0]
            successful_wp = input_pred['fourth_success'].iloc[0]
            failure_wp = input_pred['fourth_failure'].iloc[0]
            fg_prob = input_pred['fg_prob'].iloc[0]

            if recommendation_str == "Go For It":
                st.write(f"Model Recommendation: {recommendation_str}, with a {round(fourth_down_probability * 100)}% chance of converting.")
                st.write(f"Win Probability if successful: {round(successful_wp * 100, 2)}%")
                st.write(f"Win Probability if failed: {round(failure_wp * 100, 2)}%")
            elif recommendation_str == "Kick FG":
                st.write(f"Model Recommendation: {recommendation_str}, with a {round(fourth_down_probability * 100)}% chance of converting.")

with tab4:
    st.markdown("Select Weekly Recap to see the most aggressive and conservative coaching decisions. Select Coaches to see a coach's most aggressive and conservative during the season.")
    scout_mode = st.selectbox("Scout Mode", ["Weekly Recap","Coaches"])
    if scout_mode == "Weekly Recap":
        week_recap_df = base_pred_df

        @st.cache_data
        def get_years_weeks(df):
            years = sorted(df['year'].unique(), reverse=True)
            weeks_by_year = {year: sorted(df[df['year'] == year]['week'].unique()) for year in years}
            return years, weeks_by_year
        years, weeks_by_year = get_years_weeks(week_recap_df)

        selected_year = st.selectbox("Select a season", years, key="select_year")
        week_recap_df = week_recap_df[week_recap_df['year'] == selected_year]
        selected_week = st.selectbox("Select a week", weeks_by_year[selected_year])
        # Taking out anything longer than 4th and 10
        week_recap_df = week_recap_df[(week_recap_df['week'] == selected_week) & (week_recap_df['ydstogo'] < 10)]
        week_recap_df['prob_difference'] = week_recap_df.apply(model_misses, axis=1)
        top3_aggressive_plays = week_recap_df[week_recap_df['decision_class'] == 'Go For It'].sort_values(by='prob_difference', ascending=False).head(5)
        top3_conservative_plays = week_recap_df[week_recap_df['decision_class'] != 'Go For It'].sort_values(by='prob_difference', ascending=True).head(5)
        recap_text = "WEEKLY 4TH DOWN DECISION RECAP\n\nTop 3 Aggressive Plays:\n"
        for _, row in top3_aggressive_plays.iterrows():
            recap_text += scenario_sentence(row['qtr'], row['half_seconds_remaining'], row['score_differential'], row['posteam_coach'], row['posteam'], row['defteam'], row['week'], row['ydstogo'],
                        row['yardline_100'], row['decision_class'], row['model_recommendation'], row['fourth_down_probability']) + "\n"

        recap_text += "\nTop 3 Conservative Plays:\n"
        # Adjust yardline_100 to align with field yardage (i.e. OWN45) do the same for time left in the game
        for _, row in top3_conservative_plays.iterrows():
            recap_text += scenario_sentence(row['qtr'], row['half_seconds_remaining'], row['score_differential'], row['posteam_coach'], row['posteam'], row['defteam'], row['week'], row['ydstogo'],
                                          row['yardline_100'], row['decision_class'], row['model_recommendation'],
                                          row['fourth_down_probability']) + "\n"
        if st.button("See Summary"):
            st.session_state.active_tab = 3
            with st.spinner("Creating recap with OpenAI..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages = [{"role": "system", "content": ("You are an expert NFL analyst creating a concise, engaging recap of 4th down decisions for the season.\n"
                    "For each play, you will be provided:\n"
                    "- qtr (quarter)\n"
                    "- half_seconds_remaining\n"
                    "- score_differential\n"
                    "- posteam_coach (coach making the decision)\n"
                    "- posteam (team)\n"
                    "- defteam (defending team)\n"
                    "- week (game week)\n"
                    "- ydstogo (yards to go)\n"
                    "- yardline_100 (distance from opponent endzone)\n"
                    "- decision_class (what the coach decided)\n"
                    "- model_recommendation (what the model predicted)\n"
                    "- fourth_down_probability (predicted probability to go for it)\n"
                    "Instructions:\n"
                    "1. Only include plays where decision_class ≠ model_recommendation. Skip all others.\n"
                    "2. Recap the most aggressive and most conservative play calls of the week (5 plays each, but only describe the first 3 for each).\n"
                    "3. Convert half_seconds_remaining to standard clock format (e.g., 1730 → 14:30) using qtr to determine the quarter.\n"
                    "4. Indicate if the team is up, down, or tied based on score_differential.\n"
                    "5. Use yardline_100 to describe field position:\n"
                    "   - If yardline_100 > 50, the play is at the team’s own (100 - yardline_100) yard line.\n"
                    "   - If yardline_100 ≤ 50, the play is at the opponent’s yardline_100 yard line.\n"
                    "6. Include fourth_down_probability only if decision_class or model_recommendation is 'Go For It', formatted as a percentage with 1 decimal (e.g., 20.0%).\n"
                    "7. Make the recap user-friendly, with short sentences, emojis instead of numeric bullets, and NFL fan-friendly language.\n")},
                        {"role": "user", "content": recap_text}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                ai_recap = response.choices[0].message.content

            st.subheader(f"Week {selected_week} Recap")
            st.write(ai_recap)

    if scout_mode == "Coaches":
        scout_coaches_df = net_aggressive_df
        # Or select a team
        selected_coach = st.selectbox("Select a Coach", scout_coaches_df['coach'].unique())
        scout_coaches_df = scout_coaches_df[scout_coaches_df['coach'] == selected_coach]

        coach_recap_df = base_pred_df
        selected_year = st.selectbox("Select a season", sorted(coach_recap_df['year'].unique(), reverse=True), key="select_year")
        coach_recap_df = coach_recap_df[coach_recap_df['year'] == selected_year]
        # Taking out anything longer than 4th and 10
        coach_recap_df = coach_recap_df[(coach_recap_df['week'] == selected_coach) & (coach_recap_df['ydstogo'] < 10)]
        coach_recap_df['prob_difference'] = coach_recap_df.apply(model_misses, axis=1)
        top3_aggressive_plays_coach = coach_recap_df[coach_recap_df['decision_class'] == 'Go For It'].sort_values(by='prob_difference', ascending=False).head(5)
        top3_conservative_plays_coach = coach_recap_df[coach_recap_df['decision_class'] != 'Go For It'].sort_values(by='prob_difference', ascending=True).head(5)

        for _, row in scout_coaches_df.iterrows():
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; padding: 6px 0;">
                    <img src="{row['coach_url']}" style="width:100px; border-radius:8px; margin-bottom: 8px; margin-right: 12px;" />
                    <div style="font-weight: bold; font-size: 16px; margin-right: 12px;">{row['coach']}</div>
                    <div style="color: #ccc; font-size: 12px; margin-bottom: 4px; margin-right: 2px;">{row['team']}</div>
                    <img src="{row['team_logo_url']}" style="width:32px; margin-top: 2px;" />
                </div>
                """,
                unsafe_allow_html=True
            )

        recap_text = "WEEKLY 4TH DOWN DECISION RECAP\n\nTop 3 Aggressive Play Calls:\n"
        for _, row in top3_aggressive_plays_coach.iterrows():
            recap_text += scenario_sentence(row['qtr'], row['half_seconds_remaining'], row['score_differential'], row['posteam_coach'], row['posteam'], row['defteam'], row['week'], row['ydstogo'],
                        row['yardline_100'], row['decision_class'], row['model_recommendation'], row['fourth_down_probability']) + "\n"

        recap_text += "\nTop 3 Conservative Plays:\n"
        # Adjust yardline_100 to align with field yardage (i.e. OWN45) do the same for time left in the game
        for _, row in top3_conservative_plays_coach.iterrows():
            recap_text += scenario_sentence(row['qtr'], row['half_seconds_remaining'], row['score_differential'], row['posteam_coach'], row['posteam'], row['defteam'], row['week'], row['ydstogo'],
                                          row['yardline_100'], row['decision_class'], row['model_recommendation'],
                                          row['fourth_down_probability']) + "\n"
        if st.button("See Summary"):
            st.session_state.active_tab = 4
            with st.spinner("Creating recap with OpenAI..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages = [{"role": "system", "content": ("You are an expert NFL analyst creating a concise, engaging recap of 4th down decisions for the season.\n"
                    "For each play, you will be provided:\n"
                    "- qtr (quarter)\n"
                    "- half_seconds_remaining\n"
                    "- score_differential\n"
                    "- posteam_coach (coach making the decision)\n"
                    "- posteam (team)\n"
                    "- defteam (defending team)\n"
                    "- week (game week)\n"
                    "- ydstogo (yards to go)\n"
                    "- yardline_100 (distance from opponent endzone)\n"
                    "- decision_class (what the coach decided)\n"
                    "- model_recommendation (what the model predicted)\n"
                    "- fourth_down_probability (predicted probability to go for it)\n"
                    "Instructions:\n"
                    "1. Only include plays where decision_class ≠ model_recommendation. Skip all others.\n"
                    "2. Recap the most aggressive and most conservative play calls of the week (5 plays each, but only describe the first 3 for each).\n"
                    "3. Convert half_seconds_remaining to standard clock format (e.g., 1730 → 14:30) using qtr to determine the quarter.\n"
                    "4. Indicate if the team is up, down, or tied based on score_differential.\n"
                    "5. Use yardline_100 to describe field position:\n"
                    "   - If yardline_100 > 50, the play is at the team’s own (100 - yardline_100) yard line.\n"
                    "   - If yardline_100 ≤ 50, the play is at the opponent’s yardline_100 yard line.\n"
                    "6. Include fourth_down_probability only if decision_class or model_recommendation is 'Go For It', formatted as a percentage with 1 decimal (e.g., 20.0%).\n"
                    "7. Make the recap user-friendly, with short sentences, emojis instead of numeric bullets, and NFL fan-friendly language.\n")},
                        {"role": "user", "content": recap_text}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                ai_recap = response.choices[0].message.content

            st.write(ai_recap)
