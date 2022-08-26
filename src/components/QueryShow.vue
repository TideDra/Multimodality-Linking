<template>
  <div>
    <template v-for="x in segregated">
      <template v-if="typeof x === 'string'">{{ x }}</template>
      <v-chip v-else label @click="emit('query')"> {{ x.entity }} </v-chip>
    </template>
  </div>
</template>

<script setup lang="ts">
import { EntityAnswer } from "@/data";

const props = defineProps<{
  srctext: string;
  answers: EntityAnswer[];
}>();

const emit = defineEmits(["query"]);

// 对文本分组，假定每项只匹配一次
const segregated = computed(() => {
  let i = 0;
  const s = props.srctext;
  const seg: (string | EntityAnswer)[] = [];
  for (const a of props.answers) {
    const re = new RegExp(`\\b${a.entity}\\b`, "g");
    const j = s.slice(i).search(re) + i;
    seg.push(s.slice(i, j));
    seg.push(a);
    console.log(a.entity, i, j);
    i = j + a.entity.length;
  }
  seg.push(s.slice(i));
  console.log(i, s.length);
  console.log(seg);
  return seg;
});
</script>

<style lang="scss"></style>
