import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.chains import LLMChain
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
from gtts import gTTS
import pygame
import io
from janome.tokenizer import Tokenizer
from collections import Counter

# 環境変数の読み込みとクライアント初期化
load_dotenv()
client = OpenAI()

FILLER_WORDS = {
    "えー",
    "えーっと",
    "あのー",
    "そのー",
    "なんか",
    "えっと",
    "まあ",
    "こう",
}
t = Tokenizer()


def analyze_fillers(messages: list) -> dict:
    """
    会話履歴からユーザーの発言のみを抽出し、フィラーの使用状況を分析します。
    """
    user_texts = " ".join([msg["content"] for msg in messages if msg["role"] == "user"])
    if not user_texts:
        return {"total_count": 0, "details": Counter()}

    tokens = [token.surface for token in t.tokenize(user_texts)]
    filler_counts = Counter([token for token in tokens if token in FILLER_WORDS])
    total_filler_count = sum(filler_counts.values())

    return {"total_count": total_filler_count, "details": filler_counts}


def generate_feedback(conversation_history, filler_info):
    """
    会話履歴を基に、AIがフィードバックを生成します。
    """
    feedback_prompt_template = """
    あなたは、経験豊富なキャリアコンサルタントです。
    以下の就活生とAI面接官の会話履歴を厳しく分析し、プロフェッショナルな視点から詳細なフィードバックレポートを作成してください。

    # 量的分析データ
    {filler_data_summary}

    # 分析の観点
    1.  **コミュニケーション能力:** 結論から話せているか。質問の意図を正しく理解し、的確に回答できているか。
    2.  **論理性と具体性:** 回答はSTARメソッド（状況, 課題, 行動, 結果）のような構造になっているか。具体的なエピソードや数値を交えて説得力のある説明ができているか。
    3.  **自己分析の深さ:** 自身の強みや弱みを的確に捉え、それを裏付ける経験を語れているか。
    4.  **熱意とポテンシャル:** この学生から、仕事に対する熱意や将来性を感じるか。
    5.  **改善点:** もっと良くするためには、どの回答をどのように修正すればよいか。具体的な改善案を提示してください。

    # 出力形式
    必ず以下のMarkdownフォーマットで出力してください。
    - **総合評価:** 100点満点で評価し、簡単な総評を述べてください。
    - ---
    - ### 良かった点 (Good Points)
      - (箇条書きで具体的に記述)
    - ### 改善点 (Points for Improvement)
      - (箇条書きで、具体的な修正案やアドバイスを記述)
    - ### 回答の深掘り例
      - 「〇〇という回答については、△△のように深掘りすると、よりあなたの魅力が伝わります。」といった形で、具体的な深掘り例を1つ提示してください。

    # 以下が会話履歴です：
    {conversation_history}
    """

    filler_data_summary = f"- フィラー（「えーっと」「あのー」など）の使用回数: {filler_info['total_count']}回\n- フィラーの内訳: {dict(filler_info['details'])}"

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # このタスクに適したシンプルなPromptTemplateを使用
    prompt = PromptTemplate(
        template=feedback_prompt_template,
        input_variables=["conversation_history", "filler_data_summary"],
    )

    # ConversationChainの代わりにLLMChainを使用
    chain = LLMChain(llm=llm, prompt=prompt)

    # プロンプトで定義した変数名で値を渡す
    feedback = chain.predict(
        conversation_history=conversation_history,
        filler_data_summary=filler_data_summary,
    )
    return feedback


def setup_chain(difficulty):
    if difficulty == "easy":
        system_prompt = """
            あなたは、親切で優しいIT企業の採用面接官です。
            学生がリラックスして話せるように、穏やかな口調で質問をします。
            回答に詰まっても、助け舟を出すように優しくフォローしてください。
            まずは「こんにちは！リラックスしていきましょうね。まずは自己紹介をお願いします。」と挨拶してください。
            """
    elif difficulty == "normal":
        system_prompt = """
            あなたは、優秀なIT企業の採用面接官です。
            学生のポテンシャルや人柄を見抜くために、鋭い質問を投げかけます。
            あなたの目的は、ユーザー（就活生）が自己分析を深め、面接の練習ができるようにサポートすることです。
            まずは「こんにちは。本日は面接にお越しいただきありがとうございます。まずは自己紹介を1分程度でお願いします。」と挨拶と最初の質問をしてください。
            """
    elif difficulty == "hard":
        system_prompt = """
            あなたは、非常に優秀で鋭いIT企業の採用面接官です。
            回答の論理的な矛盾や具体性の欠如を厳しく追及します。
            短い時間で学生の本質を見抜くため、少し高圧的でチャレンジングな質問を投げかけてください。
            まずは「面接を始めます。自己紹介を簡潔に述べてください。」と挨拶してください。
            """

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    memory = ConversationBufferMemory(return_messages=True)
    chain = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return chain


def play_tts(text):
    fp = io.BytesIO()
    tts = gTTS(text=text, lang="ja")
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_bytes = fp.read()

    # pygameの初期化と再生
    pygame.mixer.init()
    # メモリ上のデータを直接ロード
    pygame.mixer.music.load(io.BytesIO(audio_bytes))
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


def sidebar():
    with st.sidebar:
        st.markdown("### 🔧 設定")

        if "interview_started" not in st.session_state:
            st.session_state.interview_started = False
            st.session_state.show_feedback = False

        if not st.session_state.interview_started:
            # 面接開始前の設定画面
            option = st.selectbox(
                "難易度を選んでください",
                ("easy", "normal", "hard"),
                index=None,
                placeholder="選択してください",
            )
            if st.button("開始"):
                if option:
                    st.session_state.interview_started = True
                    st.session_state.difficulty = option
                    st.rerun()
                else:
                    st.warning("難易度を設定してください")
        else:
            # 面接中の操作画面
            st.markdown(f"**難易度:** {st.session_state.difficulty}")
            st.divider()
            st.session_state.is_tts_enabled = st.toggle(
                "AIの音声を読み上げる", value=False
            )
            st.divider()
            st.markdown("### 🎤 マイクで回答")
            audio_info = mic_recorder(
                start_prompt="● 録音開始", stop_prompt="■ 録音停止", key="recorder"
            )
            st.divider()
            # 面接終了ボタンを追加
            if st.button("面接を終了する"):
                st.session_state.show_feedback = True
                st.rerun()

    return audio_info if "audio_info" in locals() else None


def get_audio_input(audio_info):
    user_input = None

    if audio_info and audio_info["bytes"]:
        if audio_info["id"] != st.session_state.get("last_audio_id"):
            st.session_state.last_audio_id = audio_info["id"]
            with st.spinner("音声を文字に変換しています..."):
                try:
                    audio_bio = ("audio.wav", audio_info["bytes"])
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", file=audio_bio
                    )
                    user_input = transcript.text
                except Exception as e:
                    st.error(f"音声認識でエラーが発生しました: {e}")

    return user_input


def get_text_input():
    return st.chat_input("質問を入力してください")


if __name__ == "__main__":
    st.set_page_config(page_title="AI面接", page_icon="🤝")
    st.title("AI面接")

    audio_info = sidebar()

    if st.session_state.get("show_feedback", False):
        st.markdown("## 📝 面接フィードバックレポート")

        with st.spinner("フィードバックを生成しています..."):
            # 会話履歴を整形
            history_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages
                ]
            )
            filler_info = analyze_fillers(st.session_state.messages)

            # フィードバックを生成
            feedback_report = generate_feedback(history_text, filler_info)
            st.markdown(feedback_report)

            with st.expander("フィラー分析詳細"):
                st.write(
                    f"**検出されたフィラーの総数:** {filler_info['total_count']} 回"
                )
                if filler_info["total_count"] > 0:
                    st.write("**内訳:**")
                    for word, count in filler_info["details"].items():
                        st.write(f"- {word}: {count} 回")
                else:
                    st.write("フィラーは検出されませんでした。素晴らしいです！")

        if st.button("もう一度面接を始める"):
            st.session_state.clear()
            st.rerun()
    elif st.session_state.get("interview_started", False):
        if "chain" not in st.session_state:
            st.session_state.chain = setup_chain(difficulty=st.session_state.difficulty)
            initial_response = st.session_state.chain.predict(input="")
            st.session_state.messages = [
                {"role": "assistant", "content": initial_response}
            ]
            st.session_state.last_audio_id = None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        text_input = get_text_input()
        audio_input = get_audio_input(audio_info)
        user_input = text_input if text_input else audio_input

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("AIが応答を考えています..."):
                    response = st.session_state.chain.predict(input=user_input)
                    st.markdown(response)
                    if st.session_state.get("is_tts_enabled", False):
                        play_tts(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
            st.rerun()
    else:
        st.markdown(
            """
            このアプリは、あなたの面接練習をサポートするAI面接官です。
            サイドバーの「設定」から難易度を選んで「開始」ボタンを押してください。
            """
        )
