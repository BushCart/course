count = 0
while (numbers := float(input())) != 0:
    if numbers >= 500:
        numbers = numbers*0.9
    count += numbers

print(count)