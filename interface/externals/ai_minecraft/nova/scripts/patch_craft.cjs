// Make Mindcraft's craftRecipe craft one at a time and confirm each craft
// with the server before it reports success.
//
// 10-26 08:07–08:15: the body "Successfully crafted wooden_pickaxe, you now
// have 1" three times at a crafting table, and each time !equip found none.
// RCON showed the server never made them: all 16 sticks were still there.
// mineflayer updates its own copy of the inventory as it clicks, and the
// server quietly rolled those clicks back; craftRecipe read the local copy
// straight away. Eva concluded the pickaxe was "a ghost" and gave up on
// stone. The same code crafted fine on a private test server (Mindcraft's
// own skills, plugins, Eva's inventory), so the trigger is something only
// her live body has; this does not guess at it.
//
// v1 waited for the server to settle, checked the item really arrived,
// tried once more, and otherwise said the craft did not go through. v2
// (10-26 12:40): bot.craft(recipe, 4) runs mineflayer's craftOnce four
// times back to back, each clicking from the local inventory copy before
// the server has confirmed the last craft. !craftRecipe("torch", 4) with
// 4 coal + 4 sticks made one craft, then threw "missing ingredient" (the
// torches arrived anyway), and the stone_pickaxe rollbacks kept coming. So
// v2 crafts once per call, settles and checks after each, retries a craft
// the server undid once, and says how many went through if not all did.
// Each rollback is logged ("[eva-world] the server rolled back crafting …").
//
// v3 (10-26 review): v2 never looked at bot.interrupt_code; Mindcraft's
// stop() kills the body process when an action ignores it for 10 s, and
// 4+ crafts at 3 s each outlast that. v3 stops before each craft and inside
// each settle, logs why a later craft threw, and words a partial success
// so the hub does not count it as a failed command.
//
// Idempotent; upgrades a v1 or v2 file; exits non-zero if the anchor is missing.
const fs = require('fs');
const file = process.argv[2];
const MARK = 'EVA-WORLD-CRAFT-CHECK v3';
const anchor = '    await bot.craft(recipe, Math.min(craftLimit.num, num), craftingTable);\n';
// exactly what the v1 patcher inserted in place of the anchor
const V1 = `    // EVA-WORLD-CRAFT-CHECK v1: the server can roll a craft back after mineflayer's own
    // inventory already shows the item; confirm it before reporting success.
    const evaSettleMs = Number(process.env.EVA_CRAFT_SETTLE_MS) || 3000;
    const evaBefore = world.getInventoryCounts(bot)[itemName] || 0;
    let evaMade = false, evaRecipe = recipe;
    for (let evaTry = 1; evaTry <= 2 && evaRecipe; evaTry++) {
        await bot.craft(evaRecipe, Math.min(craftLimit.num, num), craftingTable);
        await new Promise((r) => setTimeout(r, evaSettleMs));
        evaMade = (world.getInventoryCounts(bot)[itemName] || 0) > evaBefore;
        if (evaMade) break;
        console.log(\`[eva-world] the server rolled back crafting \${itemName} (try \${evaTry})\`);
        evaRecipe = (bot.recipesFor(mc.getItemId(itemName), null, 1, craftingTable) || [])[0];
    }
    if (!evaMade) {
        log(bot, \`Crafting \${itemName} did not go through: the server undid it, so nothing was used up.\`);
        if (placedTable) {
            await collectBlock(bot, 'crafting_table', 1);
        }
        return false;
    }
`;
// exactly what the v2 patcher inserted
const V2 = `    // EVA-WORLD-CRAFT-CHECK v2: one craft per call, each confirmed with the server, because
    // mineflayer's multi-craft clicks from a stale inventory and the server
    // can roll a craft back after the local copy already shows the item.
    const evaSettleMs = Number(process.env.EVA_CRAFT_SETTLE_MS) || 3000;
    const evaWant = Math.min(craftLimit.num, num);
    let evaDone = 0, evaUndone = 0, evaRecipe = recipe;
    while (evaDone < evaWant && evaRecipe && evaUndone < 2) {
        const evaBefore = world.getInventoryCounts(bot)[itemName] || 0;
        try {
            await bot.craft(evaRecipe, 1, craftingTable);
        } catch (err) {
            if (evaDone === 0) throw err;     // nothing made: upstream's error path
            break;
        }
        await new Promise((r) => setTimeout(r, evaSettleMs));
        if ((world.getInventoryCounts(bot)[itemName] || 0) > evaBefore) { evaDone++; evaUndone = 0; }
        else console.log(\`[eva-world] the server rolled back crafting \${itemName} (craft \${evaDone + 1}, try \${++evaUndone})\`);
        evaRecipe = (bot.recipesFor(mc.getItemId(itemName), null, 1, craftingTable) || [])[0];
    }
    if (evaDone < evaWant) {
        if (evaDone === 0) log(bot, \`Crafting \${itemName} did not go through: the server undid it, so nothing was used up.\`);
        else log(bot, \`Crafted \${itemName} \${evaDone} of \${evaWant} times; the rest did not go through. You now have \${world.getInventoryCounts(bot)[itemName] || 0} \${itemName}.\`);
        if (placedTable) {
            await collectBlock(bot, 'crafting_table', 1);
        }
        return evaDone > 0;
    }
`;
let src = fs.readFileSync(file, 'utf8');
if (src.includes(MARK)) { console.log('craft check: already applied'); process.exit(0); }
for (const old of [V1, V2]) if (src.includes(old)) src = src.replace(old, anchor);
const n = src.split(anchor).length - 1;
if (n !== 1) { console.error(`craft check: anchor found ${n} times (want 1)`); process.exit(1); }
const check = `    // ${MARK}: one craft per call, each confirmed with the server, because
    // mineflayer's multi-craft clicks from a stale inventory and the server
    // can roll a craft back after the local copy already shows the item.
    // A stop request ends it at once: Mindcraft kills the body process when
    // an action ignores one for 10 s, and n crafts take n × the settle.
    const evaSettleMs = Number(process.env.EVA_CRAFT_SETTLE_MS) || 3000;
    const evaWant = Math.min(craftLimit.num, num);
    let evaDone = 0, evaUndone = 0, evaRecipe = recipe, evaWhy = '';
    while (evaDone < evaWant && evaRecipe && evaUndone < 2 && !bot.interrupt_code) {
        const evaBefore = world.getInventoryCounts(bot)[itemName] || 0;
        try {
            await bot.craft(evaRecipe, 1, craftingTable);
        } catch (err) {
            if (evaDone === 0) throw err;     // nothing made: upstream's error path
            evaWhy = String((err && err.message) || err).replace(/^(Error: )+/, '');
            console.log(\`[eva-world] crafting \${itemName} broke off after \${evaDone}: \${evaWhy}\`);
            break;
        }
        for (let evaT = 0; evaT < evaSettleMs && !bot.interrupt_code; evaT += 250) {
            await new Promise((r) => setTimeout(r, Math.min(250, evaSettleMs - evaT)));
        }
        if ((world.getInventoryCounts(bot)[itemName] || 0) > evaBefore) { evaDone++; evaUndone = 0; }
        else if (!bot.interrupt_code) console.log(\`[eva-world] the server rolled back crafting \${itemName} (craft \${evaDone + 1}, try \${++evaUndone})\`);
        evaRecipe = (bot.recipesFor(mc.getItemId(itemName), null, 1, craftingTable) || [])[0];
    }
    if (evaDone < evaWant) {
        const evaHave = world.getInventoryCounts(bot)[itemName] || 0;
        if (bot.interrupt_code) {
            log(bot, \`Crafting \${itemName} was stopped after \${evaDone} of \${evaWant}. You now have \${evaHave} \${itemName}.\`);
            return evaDone > 0;
        }
        if (evaDone === 0) log(bot, \`Crafting \${itemName} did not go through: the server undid it, so nothing was used up.\`);
        else log(bot, \`Crafted \${itemName} \${evaDone} of \${evaWant} times; \${evaWhy ? 'then: ' + evaWhy : 'the server undid the rest'}. You now have \${evaHave} \${itemName}.\`);
        if (placedTable) {
            await collectBlock(bot, 'crafting_table', 1);
        }
        return evaDone > 0;
    }
`;
src = src.replace(anchor, check);
fs.writeFileSync(file, src);
console.log('craft check: applied');
