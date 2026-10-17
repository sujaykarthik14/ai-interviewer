import json
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

GENERATE_QUESTIONS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer. Based on the job role and experience level, generate exactly 5 relevant technical questions.
Job Role: {job_role}
Experience Level: {experience} years
Generate 5 questions that are:
1. Appropriate for the experience level
2. Technical and role-specific
3. Progressive in difficulty
4. Cover different aspects of the role
Return ONLY a JSON array of 5 questions, no explanations.
""")

EVALUATE_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer evaluating a candidate's answer.
Job Role: {job_role}
Experience Level: {experience} years
Question: {question}
Candidate's Answer: {answer}
Provide a detailed evaluation including:
1. Technical accuracy (1-10)
2. Completeness of answer (1-10)
3. Clarity of explanation (1-10)
4. Specific feedback and suggestions
5. Overall score (1-10)
Format your response as:
Score: X/10
Technical Accuracy: X/10
Completeness: X/10
Clarity: X/10
Feedback: [Your detailed feedback here]
""")

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer creating a comprehensive interview report.
Job Role: {job_role}
Experience Level: {experience} years
Interview Results:
{interview_results}
Create a professional final report including:
1. Overall assessment
2. Strengths identified
3. Areas for improvement
4. Technical competency score (average of all scores)
5. Recommendation (Pass/Fail/Consider with conditions)
6. Detailed breakdown of each question
Format as a professional report.
""")
st.title("AI Interviewer")

if "data" not in st.session_state:
    st.session_state["job_role"] = ""
    st.session_state["experience"] = 0
    st.session_state["data"] = []
    st.session_state["current_question_idx"] = 0
    st.session_state["final_report"] = ""

if st.button("Reset"):
    st.session_state.clear()
    st.rerun()

api_key = os.getenv("Gemini_API_Key")
if not api_key:
    st.info("Set Gemini_API_Key to use the app.")
    st.stop()

if not st.session_state["data"]:
    job_role = st.text_input("Job role")
    experience = st.number_input("Years of experience", min_value=0, step=1)

    if st.button("Generate questions") and job_role:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)
        prompt = GENERATE_QUESTIONS_PROMPT.format_messages(
            job_role=job_role,
            experience=experience,
        )
        questions = json.loads(llm.invoke(prompt).text)
        st.session_state["job_role"] = job_role
        st.session_state["experience"] = experience
        st.session_state["data"] = [
            {"question": question, "answer": "", "feedback": ""}
            for question in questions
        ]
        st.rerun()
else:
    data = st.session_state["data"]
    number = st.session_state["current_question_idx"]
    st.progress(number / len(data))

    if number < len(data):
        if number > 0:
            st.write("Feedback:", data[number - 1]["feedback"])

        item = data[number]
        st.write("Question", number + 1, ":", item["question"])
        answer = st.text_area("Your answer", key=f"answer_{number}")

        if st.button("Submit answer") and answer:
            llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key)
            prompt = EVALUATE_ANSWER_PROMPT.format_messages(
                job_role=st.session_state["job_role"],
                experience=st.session_state["experience"],
                question=item["question"],
                answer=answer,
            )
            item["answer"] = answer
            item["feedback"] = llm.invoke(prompt).text
            st.session_state["current_question_idx"] += 1

            if st.session_state["current_question_idx"] == len(data):
                results = ""
                for result in data:
                    results += "Question: " + result["question"] + "\n"
                    results += "Answer: " + result["answer"] + "\n"
                    results += "Feedback: " + result["feedback"] + "\n\n"
                report_prompt = FINAL_REPORT_PROMPT.format_messages(
                    job_role=st.session_state["job_role"],
                    experience=st.session_state["experience"],
                    interview_results=results,
                )
                st.session_state["final_report"] = llm.invoke(report_prompt).text
            st.rerun()
    else:
        st.write("Feedback:", data[-1]["feedback"])
        st.write(st.session_state["final_report"])
        st.download_button(
            "Download report",
            st.session_state["final_report"],
            file_name="interview_report.md",
        )
