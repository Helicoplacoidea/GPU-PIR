import subprocess

def run_test(a, b_values):
    results = []
    for b in b_values:
        # 调用 C 程序
        process = subprocess.Popen(
            ['./build/bin/test_test_fss'], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        # 输入参数到 C 程序
        input_data = f"{a} {b}\n"
        stdout, stderr = process.communicate(input=input_data)
        
        if stderr:
            print(f"Error for input ({a}, {b}): {stderr}")
        else:
            results.append(stdout.strip())  # 保存输出结果
    return results

# 定义输入数据
a_value = 20
b_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# 运行测试并打印结果
test_results = run_test(a_value, b_values)
for result in test_results:
    print(result)
