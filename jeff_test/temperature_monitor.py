import os
import platform
import subprocess
import psutil
import torch
import wmi

def get_cpu_temperature():
    """获取CPU温度"""
    system = platform.system()
    try:
        if system == "Windows":
            # 使用WMI获取CPU温度
            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            temperature_infos = w.Sensor()
            # 打印所有温度传感器用于调试
            print("\nAvailable temperature sensors:")
            for sensor in temperature_infos:
                if sensor.SensorType == 'Temperature':
                    print(f"Sensor: {sensor.Name}, Value: {sensor.Value}°C")
            
            # 尝试匹配CPU温度传感器
            cpu_temp = None
            for sensor in temperature_infos:
                if sensor.SensorType == 'Temperature' and 'Temperature #1' in sensor.Name:
                    cpu_temp = float(sensor.Value)
                    break
            
            if cpu_temp is not None:
                return cpu_temp
            
            # 如果无法获取温度，返回CPU使用率作为参考
            return psutil.cpu_percent(interval=1)
        elif system == "Linux":
            # 尝试不同的温度文件路径
            temp_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/hwmon/hwmon0/temp1_input",
                "/sys/class/hwmon/hwmon1/temp1_input"
            ]
            for path in temp_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp = float(f.read().strip()) / 1000.0
                        return temp
        elif system == "Darwin":  # macOS
            output = subprocess.check_output(['osx-cpu-temp'])
            temp = float(output.decode().strip().replace('°C', ''))
            return temp
    except Exception as e:
        print(f"无法获取CPU温度: {e}")
    return None

def get_gpu_temperature():
    """获取GPU温度"""
    try:
        if torch.cuda.is_available():
            # 在Windows上使用WMI获取GPU温度
            if platform.system() == "Windows":
                w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                temperature_infos = w.Sensor()
                for sensor in temperature_infos:
                    if sensor.SensorType == 'Temperature' and 'GPU Core' in sensor.Name:
                        return float(sensor.Value)
            
            # 如果WMI方法失败，尝试使用nvidia-smi
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'])
            temp = float(output.decode().strip())
            return temp
    except Exception as e:
        print(f"无法获取GPU温度: {e}")
    return None

def get_temperatures():
    """获取CPU和GPU温度"""
    cpu_temp = get_cpu_temperature()
    gpu_temp = get_gpu_temperature()
    
    result = {}
    if cpu_temp is not None:
        if platform.system() == "Windows" and isinstance(cpu_temp, float):
            result['cpu_temperature'] = f"{cpu_temp:.1f}°C"
        else:
            result['cpu_usage'] = f"{cpu_temp:.1f}%"
    if gpu_temp is not None:
        result['gpu_temperature'] = f"{gpu_temp:.1f}°C"
    
    return result

if __name__ == "__main__":
    temps = get_temperatures()
    for component, temp in temps.items():
        print(f"{component}: {temp}") 