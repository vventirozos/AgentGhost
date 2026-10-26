// Make Mindcraft's craftRecipe confirm a craft with the server before it
// reports success.
//
// 10-26 08:07–08:15: the body "Successfully crafted wooden_pickaxe, you now
// have 1" three times at a crafting table, and each time !equip found none.
// RCON showed the server never made them: all 16 sticks were still there.
// mineflayer updates its own copy of the inventory as it clicks, and the
// server quietly rolled those clicks back; craftRecipe read the local copy
// straight away. Eva concluded the pickaxe was "a ghost" and gave up on
// stone. The same code crafted fine on a private test server (Mindcraft's
// own skills, plugins, Eva's inventory), so the trigger is something only
// her live body has; this does not guess at it. It waits for the server to
// settle, checks the item really arrived, tries once more, and otherwise
// says the craft did not go through. Each rollback is logged
// ("[eva-world] the server rolled back crafting …") for diagnosis.
// Idempotent; exits non-zero if the anchor is missing.
const fs = require('fs');
const file = process.argv[2];
const MARK = 'EVA-WORLD-CRAFT-CHECK v1';
let src = fs.readFileSync(file, 'utf8');
if (src.includes(MARK)) { console.log('craft check: already applied'); process.exit(0); }
const anchor = '    await bot.craft(recipe, Math.min(craftLimit.num, num), craftingTable);\n';
const n = src.split(anchor).length - 1;
if (n !== 1) { console.error(`craft check: anchor found ${n} times (want 1)`); process.exit(1); }
const check = `    // ${MARK}: the server can roll a craft back after mineflayer's own
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
src = src.replace(anchor, check);
fs.writeFileSync(file, src);
console.log('craft check: applied');
