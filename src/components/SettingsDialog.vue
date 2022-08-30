<template>
  <v-dialog
    v-model="dialog"
    fullscreen
    hide-overlay
    transition="dialog-bottom-transition"
    scrollable
  >
    <v-card tile>
      <v-toolbar flat dark color="primary">
        <v-btn icon dark @click="dialog = false">
          <v-icon>mdi-close</v-icon>
        </v-btn>
        <v-toolbar-title>Settings</v-toolbar-title>
      </v-toolbar>
      <v-card-text>
        <v-container>
          <v-row>
            <v-col :cols="12" :md="4">
              <v-text-field v-model="settings.ip" label="IP" />
            </v-col>
            <v-col :cols="12" :md="4">
              <v-text-field v-model="settings.port" label="Port" />
            </v-col>
            <v-col :cols="12" :md="4">
              <v-btn @click="testAddress">测试</v-btn>
            </v-col>
          </v-row>
          <v-row>
            <v-col :cols="12" :md="4">
              <v-text-field v-model="settings.ip2" label="IP (wiki)" />
            </v-col>
            <v-col :cols="12" :md="4">
              <v-text-field v-model="settings.port2" label="Port (wiki)" />
            </v-col>
            <v-col :cols="12" :md="4">
              <v-btn>测试</v-btn>
            </v-col>
          </v-row>
        </v-container>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import axios from "axios";

const dialog = ref(false);

const settings = reactive({
  ip: "127.0.0.1",
  port: "5001",
  ip2: "127.0.0.1",
  port2: "3000",
});

const testAddress = async () => {
  const result = await axios.get(`${settings.ip}:${settings.port}/mert/test`);
  console.log(result);
};

const show = () => {
  dialog.value = true;
};

defineExpose({ show, settings });
</script>

<style lang="scss"></style>
