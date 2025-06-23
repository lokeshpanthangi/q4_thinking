import json
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_problems():
    """Load GRE math problems from the JSON file"""
    with open("problems.json", "r") as f:
        return json.load(f)

def extract_numeric_answer(response):
    """
    Extract a numeric answer from the AI's response
    This is a simplified extraction and might need adjustment based on actual responses
    """
    # Try to find numbers in the response (including decimals)
    numbers = re.findall(r'\b\d+\.\d+\b|\b\d+\b', response)
    
    if numbers:
        # Return the last number found (assuming it's the answer)
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None

def method_a(problem):
    """
    Method A: Traditional approach
    Ask AI once with low temperature to solve carefully
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Please solve this math problem very carefully with step-by-step reasoning:
    
    {problem}
    
    Solve this step-by-step and provide the final numeric answer.
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        answer_text = response.text
        
        # Extract the numeric answer
        answer = extract_numeric_answer(answer_text)
        
        return {
            "answer": answer,
            "full_response": answer_text
        }
    except Exception as e:
        print(f"Error in method_a: {e}")
        time.sleep(30)  # Wait 30 seconds if we hit rate limit
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0})
            answer_text = response.text
            
            # Extract the numeric answer
            answer = extract_numeric_answer(answer_text)
            
            return {
                "answer": answer,
                "full_response": answer_text
            }
        except Exception as e:
            print(f"Error in method_a retry: {e}")
            return {
                "answer": None,
                "full_response": f"Error: {e}"
            }

def method_b(problem, num_samples=10):  # Changed from 5 to 10 samples
    """
    Method B: Self-consistency approach
    Ask AI multiple times with high temperature and take majority vote
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Let's think step-by-step to solve this math problem:
    
    {problem}
    
    Solve this step-by-step and provide the final numeric answer.
    """
    
    answers = []
    full_responses = []
    
    for i in range(num_samples):
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 1.0})
            answer_text = response.text
            full_responses.append(answer_text)
            
            # Extract the numeric answer
            answer = extract_numeric_answer(answer_text)
            if answer is not None:
                answers.append(answer)
                
            # Add a delay to avoid hitting rate limits
            time.sleep(5)  # Wait 5 seconds between API calls
            
        except Exception as e:
            print(f"Error in method_b sample {i+1}: {e}")
            full_responses.append(f"Error: {e}")
            time.sleep(45)  # Wait longer if we hit rate limit
    
    # Get majority vote
    if answers:
        counter = Counter(answers)
        majority_vote = counter.most_common(1)[0][0]
        vote_count = {str(k): v for k, v in counter.items()}
    else:
        majority_vote = None
        vote_count = {}
    
    return {
        "majority_vote": majority_vote,
        "vote_count": vote_count,
        "all_answers": answers,
        "full_responses": full_responses
    }

def evaluate_methods():
    """
    Run both methods on all problems and compare results
    """
    problems = load_problems()
    results = []
    
    method_a_correct = 0
    method_b_correct = 0
    
    for problem_data in problems:
        problem_id = problem_data["id"]
        problem_text = problem_data["problem"]
        correct_answer = problem_data["correct_answer"]
        
        print(f"\nProblem {problem_id}:")
        print(f"{problem_text}")
        print(f"Correct answer: {correct_answer}")
        
        try:
            # Method A
            print("\nRunning Method A...")
            method_a_result = method_a(problem_text)
            method_a_answer = method_a_result["answer"]
            
            # Add a delay between methods
            time.sleep(5)
            
            # Method B
            print("\nRunning Method B...")
            method_b_result = method_b(problem_text)
            method_b_answer = method_b_result["majority_vote"]
            
            # Check accuracy (with a small tolerance for floating point)
            method_a_is_correct = abs(method_a_answer - correct_answer) < 0.01 if method_a_answer is not None else False
            method_b_is_correct = abs(method_b_answer - correct_answer) < 0.01 if method_b_answer is not None else False
            
            if method_a_is_correct:
                method_a_correct += 1
            
            if method_b_is_correct:
                method_b_correct += 1
            
            # Store results
            results.append({
                "problem_id": problem_id,
                "problem_text": problem_text,
                "correct_answer": correct_answer,
                "method_a": {
                    "answer": method_a_answer,
                    "is_correct": method_a_is_correct,
                    "full_response": method_a_result["full_response"]
                },
                "method_b": {
                    "majority_vote": method_b_answer,
                    "vote_count": method_b_result["vote_count"],
                    "all_answers": method_b_result["all_answers"],
                    "is_correct": method_b_is_correct,
                    "full_responses": method_b_result["full_responses"]
                }
            })
            
            print(f"\nMethod A answer: {method_a_answer} (Correct: {method_a_is_correct})")
            print(f"Method B majority vote: {method_b_answer} (Correct: {method_b_is_correct})")
            print(f"Method B vote distribution: {method_b_result['vote_count']}")
            
            # Save current results after each problem to a single file
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Add a delay between problems
            time.sleep(10)
            
        except Exception as e:
            print(f"Error processing problem {problem_id}: {e}")
            # Save current results to the single results file
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2)
            continue
    
    # Calculate overall accuracy
    total_problems = len(problems)
    method_a_accuracy = method_a_correct / total_problems
    method_b_accuracy = method_b_correct / total_problems
    
    print("\n===== RESULTS =====")
    print(f"Total problems: {total_problems}")
    print(f"Method A (Traditional) accuracy: {method_a_accuracy:.2%} ({method_a_correct}/{total_problems})")
    print(f"Method B (Self-Consistency) accuracy: {method_b_accuracy:.2%} ({method_b_correct}/{total_problems})")
    
    # Final results are already saved to results.json
    
    # Plot the accuracy comparison
    plot_accuracy(method_a_accuracy, method_b_accuracy)
    
    return results

def plot_accuracy(method_a_accuracy, method_b_accuracy):
    """Plot the accuracy comparison between both methods"""
    methods = ['Method A\n(Traditional)', 'Method B\n(Self-Consistency)']
    accuracies = [method_a_accuracy, method_b_accuracy]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=['blue', 'orange'])
    
    # Add accuracy values on top of the bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{accuracy:.2%}',
            ha='center',
            fontsize=12
        )
    
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title('Comparison of Method A (Traditional) vs Method B (Self-Consistency)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('accuracy.png')
    plt.close()

if __name__ == "__main__":
    evaluate_methods() 