<template>
  <v-app>
    <v-app-bar color="cyan" dense>
      <v-toolbar-title>Multimodal Entity Link - Interactive for Mert</v-toolbar-title>
    </v-app-bar>
    <v-main>
      <v-row justify="space-around">
        <v-col cols="12" md="7">
          <v-card :elevation="5" class="main-card">
            <v-subheader>输入图文</v-subheader>
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

          <v-card :elevation="5" class="main-card" style="height: 10em">
            <!-- <div v-html="answerHtml" class="answer"></div> -->
            <query-show :srctext="form.caption" :answers="answers" />
          </v-card>
        </v-col>
        <v-col cols="12" md="5">
          <v-card class="main-card" :elevation="5"> </v-card>
        </v-col>
      </v-row>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { EntityAnswer } from "@/data";
import QueryShow from "./components/QueryShow.vue";
import axios from "axios";

const form = reactive({
  imageFile: [] as File[],
  caption:
    "I'm initializing a BertForSequenceClassification model and BertForSequenceClassification model for research.",
  image: "https://picsum.photos/510/300?random",
});

const onFileChange = async () => {
  const ab = await form.imageFile[0].arrayBuffer();
  const txt =
    "data:image/png;base64," +
    window.btoa(new Uint8Array(ab).reduce((data, byte) => data + String.fromCharCode(byte), ""));
  form.image = txt;
};

const answers = ref<EntityAnswer[]>([
  {
    entity: "initializing",
    type: "LOC",
    token_ids: [1, 2],
    answer: "Q58753071",
  },
  { entity: "a", type: "LOC", token_ids: [3], answer: "Q235670" },
  {
    entity: "BertForSequenceClassification",
    type: "LOC",
    token_ids: [4, 5, 6, 7, 8, 9, 10],
  },
  { entity: "model", type: "LOC", token_ids: [11], answer: "Q486902" },
  {
    entity: "BertForSequenceClassification",
    type: "LOC",
    token_ids: [12, 13, 14, 15, 16, 17, 18],
  },
  { entity: "model", type: "LOC", token_ids: [19], answer: "Q1979154" },
]);

const submit = async () => {
  const result = JSON.parse(
    await axios.post("http://175.27.209.2:5001/mert/query", {
      image: form.image,
      caption: form.caption,
    })
  );
  console.log(result);
  const result2 = await axios.post("http://127.0.0.1:3002/mert/wiki", {
    data: JSON.stringify(result.query),
  });
  console.log(result2);
  const result3 = await axios.post("http://175.27.209.2:5001/mert/back", {
    key: result.key,
    data: result2,
  });
  console.log(result3);
  //answers.value = JSON.parse(result3);
};
</script>

<style lang="scss" scoped>
.main-card {
  margin: 1em;
  padding: 1em;
}

.entity-tag {
  font-weight: bold;
  border: 2px solid red;
}
</style>