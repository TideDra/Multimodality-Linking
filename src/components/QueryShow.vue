<template>
  <div>
    <template v-for="x in segregated">
      <template v-if="typeof x === 'string'">{{ x }}</template>
      <v-chip v-else label @click="emit('query', x)" :color="chipColor(x)">
        {{ x.entity }}
      </v-chip>
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

const chipColor = (entity: EntityAnswer) => {
  console.log(entity.type);
  switch (entity.type) {
    case "ORG":
      return "#BA68C8";
    case "PER":
      return "#4FC3F7";
    case "LOC":
      return "#81C784";
    case "MISC":
      return "#A1887F";
    default:
      return "#BDBDBD";
  }
};
</script>

<style lang="scss">
.v-chip.v-chip--size-default {
  --v-chip-height: 2em !important;
  padding: 0 4px !important;
}
</style>