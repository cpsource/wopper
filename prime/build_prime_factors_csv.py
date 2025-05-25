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

def number_of_factors(n):
    """Return the number of distinct prime factors of n."""
    count = 0
    first_factor = False
    i = 2
    while i * i <= n:
        if n % i == 0 and is_prime(i):
            count += 1
            while n % i == 0:
                n //= i
        i += 1
    if n > 1 and is_prime(n):
        count += 1
        if not first_factor:
            first_factor = n
            
    return count, n

def build_csv(filename,test_filename):
    """Build the CSV file with required prime and two-prime-factor values."""
    result = []
    result_test = []
    first_prime = None
    last_prime = None
    cnt = 0
    tst = False
    
    # walk the walk
    for num in range(LOWER, UPPER + 1, 2):
        cnt += 1
        if cnt % 10000 == 0:
            print(f"cnt = {cnt}")
            tst = True
        else:
            tst = False
        if is_prime(num):
            if tst:
                result_test.append([num, 0])
            else:
                result.append([num, 0])
            if not first_prime:
                first_prime = num;
            else:
                if not last_prime:
                    last_prime = num;

            if first_prime and last_prime:
                # Find an odd non-prime number with exactly two distinct prime factors
                for xnum in range(first_prime + 2, last_prime, 2):
                    numfac, firstfac = number_of_factors(xnum)
                    if numfac == 2:
                        if tst:
                            result_test.append([xnum, firstfac])
                        else:
                            result.append([xnum, firstfac])
                        break
                first_prime = last_prime;
                last_prime = num

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

