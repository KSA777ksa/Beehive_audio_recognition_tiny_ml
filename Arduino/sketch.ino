#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Adafruit_ZeroFFT.h>
#include <HardwareSerial.h>

// Конфигурация
constexpr int AUDIO_DURATION = 3;
constexpr int SAMPLE_RATE = 16000;
constexpr int NUM_SAMPLES = AUDIO_DURATION * SAMPLE_RATE;
constexpr int N_FFT = 2048;
constexpr int HOP_LENGTH = 160;
constexpr int N_MELS = 128;
constexpr int N_MFCC = 13;
constexpr int MFCC_FRAMES = 300;
constexpr int TENSOR_ARENA_SIZE = 40 * 1024;

// Глобальные параметры из обучения
const float GLOBAL_MEAN = 6.025622f;
const float GLOBAL_STD = 47.826233f;

// Аппаратные настройки
const int MIC_PIN = A0;
const int GSM_TX = 0;
const int GSM_RX = 1;
const int BUTTON_PIN = 9;

alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Предрассчитанный банк мель-фильтров
#include "mel_filters.h"

// DCT матрица для 13 коэффициентов
const float dct_matrix[13][40] = {
  {0.111, 0.157}
};

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.begin(115200);
  Serial1.begin(115200, SERIAL_8N1, GSM_TX, GSM_RX);
  
  analogReadResolution(12);
  analogSetPinAttenuation(MIC_PIN, ADC_11db);
  
  // Инициализация модели
  model = tflite::GetModel("bee_detector.tflite");
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    while (1) Serial.println("Ошибка аллокации тензоров!");
  }
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Калибровка микрофона
  calibrate_mic();
}

void calibrate_mic() {
  // 10ms Калибровка для определения смещения
  long sum = 0;
  for (int i = 0; i < 100; i++) {
    sum += analogRead(MIC_PIN);
    delayMicroseconds(100);
  }
  adc_offset = sum / 100;
}

void apply_window(float* data, int size) {
  // Оптимизированное применение окна Ханна
  const float a = 2 * M_PI / (size - 1);
  for (int i = 0; i < size; i++) {
    float multiplier = 0.5 - 0.5 * cos(a * i);
    data[i] *= multiplier;
  }
}

void compute_mfcc(float* audio, float* mfcc_out) {
  // 1. Преэмпфазис
  for (int i = NUM_SAMPLES - 1; i > 0; i--) {
    audio[i] -= 0.97 * audio[i - 1];
  }
  
  // 2. Вычисление спектрограммы
  int num_frames = (NUM_SAMPLES - N_FFT) / HOP_LENGTH + 1;
  float frame[N_FFT];
  float power_spectrum[N_FFT / 2];
  
  for (int f = 0; f < num_frames; f++) {
    // Извлечение фрейма
    int offset = f * HOP_LENGTH;
    memcpy(frame, audio + offset, N_FFT * sizeof(float));
    
    // Применение окна
    apply_window(frame, N_FFT);
    
    // FFT
    ZeroFFT(frame, N_FFT);
    
    // Спектр мощности
    for (int i = 0; i < N_FFT / 2; i++) {
      float real = frame[2 * i];
      float imag = frame[2 * i + 1];
      power_spectrum[i] = real * real + imag * imag;
    }
    
    // Применение мель-фильтров
    float mels[N_MELS] = {0};
    for (int m = 0; m < N_MELS; m++) {
      for (int j = 0; j < mel_filter_lengths[m]; j++) {
        int idx = mel_filter_indices[m][j];
        mels[m] += power_spectrum[idx] * mel_filter_weights[m][j];
      }
      mels[m] = logf(mels[m] + 1e-6);
    }
    
    // DCT (MFCC)
    for (int c = 0; c < N_MFCC; c++) {
      float sum = 0;
      for (int m = 0; m < N_MELS; m++) {
        sum += mels[m] * dct_matrix[c][m];
      }
      mfcc_out[f * N_MFCC + c] = sum;
    }
  }
  
  // Дополнение до 300 фреймов
  for (int i = num_frames * N_MFCC; i < MFCC_FRAMES * N_MFCC; i++) {
    mfcc_out[i] = 0.0f;
  }
}

void send_sms_alert(float confidence) {
  Serial1.println("AT+CMGF=1");
  delay(100);
  Serial1.println("AT+CMGS=\"+79123456789\"");
  delay(100);
  Serial1.print("Пчелы обнаружены! Уверенность: ");
  Serial1.print(confidence * 100);
  Serial1.println("%");
  Serial1.write(26);  // Ctrl+Z
}

void loop() {
  // Ожидание кнопки для нового измерения
  if (digitalRead(BUTTON_PIN) == HIGH) {
    return;
  }
  
  // Запись аудио
  float audio_buffer[NUM_SAMPLES];
  unsigned long start_time = millis();
  
  for (int i = 0; i < NUM_SAMPLES; i++) {
    int sample = analogRead(MIC_PIN) - adc_offset;
    audio_buffer[i] = sample * (3.3 / 4096.0);  // Преобразование в напряжение
    while (millis() - start_time < i * 1000.0 / SAMPLE_RATE);
  }
  
  // Вычисление MFCC
  float mfcc_buffer[MFCC_FRAMES * N_MFCC];
  compute_mfcc(audio_buffer, mfcc_buffer);
  
  // Нормализация и квантование
  for (int i = 0; i < MFCC_FRAMES * N_MFCC; i++) {
    float normalized = (mfcc_buffer[i] - GLOBAL_MEAN) / GLOBAL_STD;
    input->data.int8[i] = static_cast<int8_t>(normalized / input->params.scale + input->params.zero_point);
  }
  
  // Инференс
  if (interpreter->Invoke() != kTfLiteOk) {
    return;
  }
  
  // Деквантование выхода
  float bee_prob = (output->data.int8[1] - output->params.zero_point) * output->params.scale;
  
  // Отправка оповещения
  if (bee_prob > 0.8) {
    send_sms_alert(bee_prob);
  }
  
  // Задержка перед следующим измерением
  delay(30000);
}