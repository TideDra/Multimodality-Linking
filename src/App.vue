<template>
  <v-app>
    <v-app-bar color="cyan" dense>
      <v-toolbar-title>Multimodal Entity Link - Interactive for Mert</v-toolbar-title>
      <v-spacer />
      <v-btn icon @click="settingsDialog.show()">
        <v-icon>mdi-dots-vertical</v-icon>
      </v-btn>
    </v-app-bar>
    <v-main>
      <v-row justify="space-around">
        <v-col cols="12" md="8">
          <v-card :elevation="5" class="main-card">
            <v-card-subtitle>输入图文</v-card-subtitle>
            <v-form>
              <v-row justify="space-around">
                <v-col :cols="7">
                  <v-file-input
                    v-model="form.imageFile"
                    label="Image input"
                    accept="image/*"
                    show-size
                    clearable
                    @change="onFileChange"
                  />
                  <v-textarea v-model="form.caption" outlined label="Caption" />
                </v-col>
                <v-col :cols="5">
                  <v-img :src="form.image" contain />
                </v-col>
              </v-row>
              <v-btn color="primary" @click="submit">查询</v-btn>
            </v-form>
          </v-card>
        </v-col>
        <v-col cols="12" md="4">
          <v-card :elevation="5" class="main-card" style="height: 90%">
            <v-card-subtitle>Recognized Entities</v-card-subtitle>
            <query-show :srctext="form.caption" :answers="answers" @query="onQuery" />
          </v-card>
        </v-col>
      </v-row>
      <!-- <v-row style="width: 100%"> -->
      <v-card class="main-card" :elevation="5" style="height: 100%; padding: 1px; margin-top: 1em">
        <iframe :src="wikiSrc" style="width: 100%; height: 100%"></iframe>
      </v-card>
      <!-- </v-row> -->
      <settings-dialog ref="settingsDialog" />
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { EntityAnswer } from "@/data";
import stub from "@/data/stub";
import QueryShow from "./components/QueryShow.vue";
import axios from "axios";
import SettingsDialog from "./components/SettingsDialog.vue";

const form = reactive({
  imageFile: [] as File[],
  caption: stub.caption,
  image: "https://picsum.photos/510/300?random",
});

const settingsDialog = ref();

const onFileChange = async () => {
  const ab = await form.imageFile[0].arrayBuffer();
  const txt =
    "data:image/png;base64," +
    window.btoa(new Uint8Array(ab).reduce((data, byte) => data + String.fromCharCode(byte), ""));
  form.image = txt;
};

const wikiSrc = ref("https://www.wikidata.org/wiki/Wikidata:Main_Page");

const answers = ref<EntityAnswer[]>(stub.answer);
const wikidata = ref(stub.wikidata);

const submit = async () => {
  const formatUrl = (addr: string, port: string, post: string) => `http://${addr}:${port}/${post}`;
  const settings = settingsDialog.value.settings;
  const { data: result } = await axios.post(formatUrl(settings.ip, settings.port, "mert/query"), {
    image: form.image,
    caption: form.caption,
  });
  console.log(result);
  if (result.query) {
    const { data: result2 } = await axios.post(
      formatUrl(settings.ip2, settings.port2, "mert/wiki"),
      {
        data: result.query,
      }
    );
    console.log(result2);
    const { data: result3 } = await axios.post(formatUrl(settings.ip, settings.port, "mert/back"), {
      key: result.key,
      data: result2,
    });
    console.log(result3);
    showResult(result3);
  } else {
    showResult(result);
  }
};

const showResult = (obj: any) => {
  console.log(obj);
  answers.value = obj.answer;
  wikidata.value = obj.wikidata;
};

const onQuery = (entity: EntityAnswer) => {
  if (entity.answer) {
    wikiSrc.value = "https://www.wikidata.org/entity/" + entity.answer;
  }
  console.log(entity);
};
</script>

<style lang="scss" scoped>
.v-main {
  padding: 1em;
  padding-top: calc(var(--v-layout-top) + 1em);
}

.main-card {
  padding: 1em;
}

.entity-tag {
  font-weight: bold;
  border: 2px solid red;
}
</style>