import re


class RewardManager:
    def __init__(self, tokenizer, reasoning_end, solution_start, solution_end):
        self.tokenizer = tokenizer
        self.reasoning_end = reasoning_end
        self.solution_start = solution_start
        self.solution_end = solution_end

        # Regex setup
        solution_end_regex = (
            r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
        )
        self.match_format = re.compile(
            rf"{reasoning_end}.*?"
            rf"{solution_start}(.+?){solution_end_regex}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )

        self.match_numbers = re.compile(
            solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL,
        )

        self.printed_times = 0
        self.print_every_steps = 5

    def match_format_exactly(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if self.match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score += 0.5 if response.count(self.reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(self.solution_start) == 1 else -1.0
            score += 0.5 if response.count(self.solution_end) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None
            else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.0)
                continue

            if guess == true_answer:
                scores.append(5.0)
            elif guess.strip() == true_answer.strip():
                scores.append(3.5)
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        scores.append(2.0)
                    elif 0.8 <= ratio <= 1.2:
                        scores.append(1.5)
                    else:
                        scores.append(-2.5)
                except (ValueError, ZeroDivisionError):
                    scores.append(-4.5)
        return scores

    def check_numbers(self, prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None
            else None
            for r in responses
        ]

        # Debug printing
        if self.printed_times % self.print_every_steps == 0:
            print(
                f"{'*' * 20}\nQuestion:\n{question}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}"
            )
        self.printed_times += 1

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                true_val = float(true_answer.strip())
                guess_val = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess_val == true_val else -1.5)
            except (ValueError, ZeroDivisionError):
                scores.append(0.0)
        return scores

    def get_reward_funcs(self):
        return [
            self.match_format_exactly,
            self.match_format_approximately,
            self.check_answer,
            self.check_numbers,
        ]
