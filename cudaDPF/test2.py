import subprocess

def run_test(a_values, b):
    results = []
    for a in a_values:
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
a_values = [19,20,21,22,23,24]
b_value = 1

# 运行测试并打印结果
test_results = run_test(a_values, b_value)
for result in test_results:
    print(result)
