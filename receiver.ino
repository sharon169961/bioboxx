#include <esp_now.h>
#include <WiFi.h>
#include <ArduinoJson.h>

typedef struct struct_message {
  char node_id[10];
  float temp;
  float ph;
} struct_message;

struct_message incomingData;

// ✅ NEW ESP-NOW CALLBACK
void OnDataRecv(const esp_now_recv_info *info, const uint8_t *incomingDataRaw, int len) {
  memcpy(&incomingData, incomingDataRaw, sizeof(incomingData));

  StaticJsonDocument<200> doc;

  doc["node_id"] = incomingData.node_id;
  doc["temp"] = incomingData.temp;
  doc["ph"] = incomingData.ph;

  serializeJson(doc, Serial);
  Serial.println();
}


void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();  // 🔥 IMPORTANT

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW Init Failed");
    return;
  }

  esp_now_register_recv_cb(OnDataRecv);

  Serial.println("✅ Gateway Ready");
}

void loop() {}