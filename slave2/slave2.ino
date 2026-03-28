#include <esp_now.h>
#include <WiFi.h>
#include <string.h>

uint8_t receiverAddress[] = {0xF0, 0x24, 0xF9, 0x0E, 0x75, 0xFC};

typedef struct struct_message {
  char node_id[10];
  float temp;
  float ph;
} struct_message;

struct_message data;

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  randomSeed(esp_random());          // 🔥 important
  delay(random(0, 3000));            // 🔥 stagger startup

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW Init Failed");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  esp_now_add_peer(&peerInfo);

  Serial.println("Node Ready");
}

void loop() {
  strcpy(data.node_id, "reef_02");

  data.temp = random(25, 28);
  data.ph = random(78, 82) / 10.0;

  esp_err_t result = esp_now_send(receiverAddress, (uint8_t *)&data, sizeof(data));

  if (result == ESP_OK) {
    Serial.println("Send Success");
  } else {
    Serial.println("Send Failed");
  }

  delay(2000 + random(0, 2000));   // 🔥 desync transmissions
}

