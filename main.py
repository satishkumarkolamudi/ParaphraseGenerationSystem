from cpg import custom_paraphrase
from llm_paraphraser import llm_paraphrase
from metrics import *

input_text = '''
A cover letter is a formal document that accompanies your resume when you apply for a job. It serves as an introduction and provides additional context for your application. Here’s a breakdown of its various aspects: Purpose The primary purpose of a cover letter is to introduce yourself to the hiring manager and to provide context for your resume. It allows you to elaborate on your qualifications, skills, and experiences in a way that your resume may not fully capture. It’s also an opportunity to express your enthusiasm for the role and the company, and to explain why you would be a good fit. Content A typical cover letter includes the following sections:
1. Header: Includes your contact information, the date, and the employer’s contact information.
2. Salutation: A greeting to the hiring manager, preferably personalized with their name.
3. Introduction: Briefly introduces who you are and the position you’re applying for.
4. Body: This is the core of your cover letter where you discuss your qualifications, experiences, and skills that make you suitable for the job. You can also mention how you can contribute to the company.
5. Conclusion: Summarizes your points and reiterates your enthusiasm for the role. You can also include a call to action, like asking for an interview.
6. Signature: A polite closing (“Sincerely,” “Best regards,” etc.) followed by your name. Significance in the Job Application Process The cover letter is often the first document that a hiring manager will read, so it sets the tone for your entire application. It provides you with a chance to stand out among other applicants and to make a strong first impression. Some employers specifically require a cover letter, and failing to include one could result in your application being disregarded. In summary, a cover letter is an essential component of a job application that serves to introduce you, elaborate on your qualifications, and make a compelling case for why you should be considered for the position.
'''

# CPG
cpg_output, cpg_time = measure_latency(custom_paraphrase, input_text)

# LLM
llm_output, llm_time = measure_latency(llm_paraphrase, input_text)

print("\n===== LENGTH CHECK =====")
print("Input words:", len(input_text.split()))
print("CPG words:", len(cpg_output.split()))
print("LLM words:", len(llm_output.split()))

print("\n===== Output Text =====")
print("CPG Output:", cpg_output)
print("LLM Output:", llm_output)

print("\n===== QUALITY METRICS =====")
print("CPG Semantic Similarity:", semantic_similarity(input_text, cpg_output))
print("LLM Semantic Similarity:", semantic_similarity(input_text, llm_output))

print("CPG BLEU:", bleu(input_text, cpg_output))
print("LLM BLEU:", bleu(input_text, llm_output))

print("CPG Readability:", readability(cpg_output))
print("LLM Readability:", readability(llm_output))

print("\n===== LATENCY (ms) =====")
print("CPG Latency:", round(cpg_time, 4))
print("LLM Latency:", round(llm_time, 4))
