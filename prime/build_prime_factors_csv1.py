import csv
import math

# Define range
LOWER = 3
UPPER = 2**20 - 1

def is_prime(n):
    """Check if n is a prime number."""
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def get_factors(n):
    """Return a list of all factors of n."""
    return [i for i in range(3, n) if n % i == 0]

def build_csv(filename,test_filename):
    """Build the CSV file with required prime and two-prime-factor values."""
    result = []
    result_test = []
    tst = False
    cnt = 0
    
    # walk the walk
    for num in range(LOWER, UPPER + 1, 2):
        if not is_prime(num):
            factors = get_factors(num)
            count = len(factors)

            #print(f"factors = {factors}")
            
            if not count == 2 :
                continue
            if not is_prime(factors[0]) :
                continue
            if not is_prime(factors[1]) :
                continue

            #print(f"cnt = {cnt}, num = {num}")

            cnt += 1
            if cnt % 100 == 0:
                print(f"cnt = {cnt}")
                tst = True
            else:
                tst = False
            
            if tst:
                result_test.append([num, max(factors[0],factors[1])])
            else:
                result.append([num, max(factors[0],factors[1])])

            if cnt >= 10000:
                break
            
    # Write to CSV
    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Number", "First Prime"])
        writer.writerows(result)

    # Write to CSV
    with open(test_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Number", "First Prime"])
        writer.writerows(result_test)
        
    print(f"✅ CSV file '{filename}' has been created.")

# Run the program
if __name__ == "__main__":
    build_csv("prime_factors_output.csv", "test_factors_output.csv")

