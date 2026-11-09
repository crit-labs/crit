"""The fixed texts the smoke tests send. They are written for these tests and share nothing with crit-eval's
prompts or held-out text, so no engine choice can be fitted to Eval through them."""
import random

# Plain prose of about 700 tokens on most tokenizers: long enough for prefixes past 512 tokens, ordinary enough
# that any working language model reads it at a low perplexity.
PROSE = """The old harbour town sits where the river meets the sea, and for most of its history it lived on what the \
boats brought in. In the mornings the fishermen sorted their catch on the stone quay while gulls circled overhead, \
and by noon the market square was full of people buying bread, vegetables and fish for the evening meal. Children \
ran between the stalls, and the older residents sat on the benches outside the church, talking about the weather \
and the price of everything.

Over the last fifty years the town has changed. The fishing fleet is smaller now, and many of the warehouses along \
the water have been turned into apartments, cafes and small shops. Tourists arrive in the summer by train and by car, \
and the population nearly doubles between June and September. Some residents welcome the visitors because they bring \
money and keep the restaurants open. Others complain that the streets are crowded, that rents have risen, and that \
young families can no longer afford to live in the houses where their grandparents grew up.

The town council has tried to find a balance. It has built a new school on the edge of town, repaired the sea wall \
after a winter storm, and set aside a few streets where only residents may park. It also supports a small museum \
that tells the story of the harbour, with photographs of the great flood, models of the wooden boats that were once \
built in the yard by the lighthouse, and letters written by sailors to their families during long voyages.

A visitor who walks along the coast path in the early evening will see the town at its best. The light turns gold \
over the water, the boats rock gently at their moorings, and the smell of cooking drifts from open windows. From the \
top of the hill beyond the lighthouse one can see the whole bay, the river winding inland between green fields, and \
on a clear day the outline of the islands far to the west.

Scientists who study the coast say that the sea level is rising slowly, and that within a century parts of the lower \
town may flood more often. Engineers have proposed several solutions: a higher wall, a barrier across the mouth of \
the river, or a gradual move of the most exposed buildings to higher ground. Each plan has its costs and its \
supporters, and the debate in the council chamber is often long and heated. What everyone agrees on is that the town \
must prepare, because the water will not wait for a decision.

In the meantime, life goes on much as before. The baker opens his shop at six, the ferry leaves for the islands at \
nine, and every Saturday the band plays in the square. People who moved away often come back for the summer festival, \
when the boats are decorated with flags and the whole town gathers by the water to watch the fireworks. For a few \
hours, the arguments about rents and walls are forgotten, and the harbour belongs to everyone again."""

# Common words for the recall texts: a random sequence of them, written twice, which a working model reads the
# second time at a very low loss by copying from the first.
WORDS = """apple river stone window garden silver morning yellow bridge forest candle mountain paper engine orange
basket dinner summer winter pocket hammer letter mirror island button ladder pencil rabbit violin wagon anchor
blanket castle desert feather guitar harbor jacket kettle lemon marble needle ocean pepper quilt rocket saddle
tunnel umbrella valley whistle zipper bottle carpet dragon finger glove helmet insect jungle kitchen lantern
meadow nickel orchard parrot rubber shovel tiger velvet walnut acorn bucket cherry donkey eagle fountain goose
honey igloo jelly koala lizard magnet napkin otter pillow quiver raisin spider tomato urchin vessel wizard
barrel cactus diamond falcon ginger hollow ribbon sandal timber copper meteor planet comet glacier canyon prairie
lagoon volcano tundra pebble boulder cavern summit""".split()

PROMPTS = (  # chat turns for the greedy-reply tests
    "Write two short sentences about rivers.",
    "List three colours and one fruit for each.",
    "Explain in one sentence what a bicycle is.",
    "Name four animals that live in the sea.",
)


def recall_words(count, seed=7):
    """`count` common words in a fixed random order, as one line of text."""
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(count))
