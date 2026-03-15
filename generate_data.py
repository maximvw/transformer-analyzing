"""
Генерация данных для 4x4 digit multiplication в формате Internalize_CoT_Step_by_Step.

Формат каждой строки:
  <input>||<CoT> #### <answer>

Где:
- input: "a0 a1 a2 a3 * b0 b1 b2 b3" (цифры reversed, least significant first)
- CoT: partial products и running sums
- answer: цифры произведения (reversed)

Пример: 2365 * 4347
  Input:  5 6 3 2 * 7 4 3 4
  CoT:    5 5 5 6 1 + 0 0 6 4 9 0 ( 5 5 1 1 1 1 ) + 0 0 5 9 0 7 0 ( 5 5 6 0 2 8 0 ) + 0 0 0 0 6 4 9 0
  Answer: 5 5 6 0 8 2 0 1
"""

import random
import argparse
import os


def number_to_reversed_digits(n: int, num_digits: int) -> list[int]:
    """Конвертирует число в список цифр (reversed, least significant first), с padding нулями."""
    digits = []
    for _ in range(num_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def digits_to_str(digits: list[int]) -> str:
    """Список цифр в строку через пробел."""
    return " ".join(str(d) for d in digits)


def generate_cot(a: int, b: int, D: int = 4) -> str:
    """
    Генерирует Chain-of-Thought для a * b.

    Формат: p0 + p1 (sum01) + p2 (sum012) + p3
    где p_i = a * b_i * 10^i
    """
    b_digits = number_to_reversed_digits(b, D)

    parts = []
    running_sum = 0

    for i in range(D):
        partial = a * b_digits[i] * (10 ** i)
        # Partial product имеет D + 1 + i цифр
        num_digits_partial = D + 1 + i
        partial_str = digits_to_str(number_to_reversed_digits(partial, num_digits_partial))

        running_sum += partial

        if i == 0:
            parts.append(partial_str)
        elif i == D - 1:
            # Последний partial product — без running sum в скобках после него
            parts.append(f"{partial_str}")
        else:
            # Промежуточные: partial product + running sum в скобках
            num_digits_sum = D + 1 + i
            sum_str = digits_to_str(number_to_reversed_digits(running_sum, num_digits_sum))
            parts.append(f"{partial_str} ( {sum_str} )")

    return " + ".join(parts)


def generate_example(a: int, b: int, D: int = 4) -> str:
    """Генерирует одну строку данных."""
    product = a * b

    a_str = digits_to_str(number_to_reversed_digits(a, D))
    b_str = digits_to_str(number_to_reversed_digits(b, D))

    cot = generate_cot(a, b, D)
    answer = digits_to_str(number_to_reversed_digits(product, 2 * D))

    return f"{a_str} * {b_str}||{cot} #### {answer}"


def main():
    parser = argparse.ArgumentParser(description="Generate multiplication training data")
    parser.add_argument("--D", type=int, default=4, help="Number of digits (default: 4)")
    parser.add_argument("--train_size", type=int, default=8000, help="Training set size")
    parser.add_argument("--val_size", type=int, default=1000, help="Validation set size")
    parser.add_argument("--test_size", type=int, default=1000, help="Test set size")
    parser.add_argument("--output_dir", type=str,
                        default="Internalize_CoT_Step_by_Step/data/4_by_4_mult",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    D = args.D
    min_val = 10 ** (D - 1)  # 1000 для D=4
    max_val = 10 ** D - 1     # 9999 для D=4

    total = args.train_size + args.val_size + args.test_size

    # Генерируем уникальные пары
    pairs = set()
    while len(pairs) < total:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        pairs.add((a, b))

    pairs = list(pairs)
    random.shuffle(pairs)

    train_pairs = pairs[:args.train_size]
    val_pairs = pairs[args.train_size:args.train_size + args.val_size]
    test_pairs = pairs[args.train_size + args.val_size:]

    os.makedirs(args.output_dir, exist_ok=True)

    for filename, data_pairs in [
        ("train.txt", train_pairs),
        ("valid.txt", val_pairs),
        ("test_bigbench.txt", test_pairs),
    ]:
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w") as f:
            for a, b in data_pairs:
                f.write(generate_example(a, b, D) + "\n")
        print(f"Written {len(data_pairs)} examples to {filepath}")


if __name__ == "__main__":
    main()
