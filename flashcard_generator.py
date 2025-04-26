''' Tool to assist in creating Japanese flashcards.'''

import concurrent.futures
import json
import os
import random

from anki.collection import Collection
from anki.notes import Note
from openai import OpenAI

# New words to create flashcards for, one per line. Lines beginning with # are comments and will be ignored.
# Blank lines are also ignored, as is anything following a separating whitespace character on a line.
# These words will be de-duped and added to the pending words text file in random order when the tool runs.
_NEW_WORDS = '''
'''

# The role for the LLM to take when being prompted.
_SYSTEM_ROLE = ("You are a Japanese language assistant, skilled in coming up with useful example sentences for "
                "JSON-formatted flashcards.")

# Path to the Anki collection file.
_COLLECTION_PATH = "~/Library/Application Support/Anki2/Eam/collection.anki2"

# Anki deck name and note types (models) to use.
_DECK_NAME = "Japanese"
_CLOZE_MODEL_NAME = "Japanese Cloze"
_BASIC_MODEL_NAME = "Japanese Reading"

# Note field names.
_CLOZE_EXAMPLE_SENTENCE_FIELD = "Expression"
_CLOZE_ENGLISH_FIELD = "Extra"
_BASIC_JAPANESE_FIELD = "Expression"
_BASIC_ENGLISH_FIELD = "Meaning"

# Files containing model prompts.
_CLOZE_PROMPT_PATH = "cloze_prompt.txt"
_BASIC_PROMPT_PATH = "basic_prompt.txt"

# Word list file paths.
_WORDS_PENDING_PATH = "words_pending.txt"
_WORDS_DONE_PATH = "words_done.txt"
_WORDS_SKIPPED_PATH = "words_skipped.txt"


def is_kana(s):
  """Check if a string consists only of hiragana or katakana characters."""
  return all("\u3040" <= c <= "\u309F" or "\u30A0" <= c <= "\u30FF" for c in s)


def _get_new_words_from_multiline_string(word_string: str) -> list[str]:
  lines = word_string.split("\n")
  # Some lines have a translation or reading after the word, so we only want the first word. Also skip comments.
  words = [line.strip().split()[0] for line in lines if line.strip() and not line.strip().startswith("#")]
  # Remove duplicates.
  words = list(set(words))
  # Randomize the order of the words.
  random.shuffle(words)
  return words


def _load_pending_words_from_file() -> list[str]:
  with open(_WORDS_PENDING_PATH, "r", encoding="utf-8") as f:
    words = f.read().split("\n")
  return words


def _append_new_pending_words_to_file(new_words: list[str], existing_words: list[str]):
  with open(_WORDS_PENDING_PATH, "a", encoding="utf-8") as f:
    for word in new_words:
      if word not in existing_words:
        print(f"Adding to pending words file: {word}")
        f.write(word + "\n")


def _request_flashcards_from_llm(word: str):
  prompt_path = _BASIC_PROMPT_PATH if is_kana(word) else _CLOZE_PROMPT_PATH

  # Load prompt from file.
  with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()
  # Replace the placeholder with the word.
  prompt = prompt.replace("{{WORD}}", word)

  # Send the prompt to the API.
  client = OpenAI()
  completion = client.chat.completions.create(model="gpt-4o",
                                              response_format={"type": "json_object"},
                                              messages=[{
                                                  "role": "system",
                                                  "content": _SYSTEM_ROLE
                                              }, {
                                                  "role": "user",
                                                  "content": prompt
                                              }])
  return completion


def _remove_word_from_pending_file(word: str):
  with open(_WORDS_PENDING_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
  lines = [line for line in lines if line.strip() != word]
  with open(_WORDS_PENDING_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)


def _create_flashcard_interactively(word: str, completion, anki_collection: Collection, deck_id, cloze_model_id,
                                    basic_model_id):
  print(f"Flashcard options: {word}\n")
  choice = completion.choices[0]

  # Parse JSON response.
  response = json.loads(choice.message.content)
  flashcards = response["flashcards"]
  for i, card in enumerate(flashcards):
    print(f"{i + 1}   {card['example_sentence']}")  # pylint: disable=inconsistent-quotes
    print(f"    {card['english']}\n")  # pylint: disable=inconsistent-quotes

  # Get user input string.
  while True:
    user_input = input("Enter \"1\" to keep first card, \"2\" to keep second, \"13\" to keep first and third, etc. " +
                       "Blank to skip this word without creating any cards.\nFlashcards to keep: ")

    # Skip word if no input.
    if not user_input:
      # Append word to skipped file.
      with open(_WORDS_SKIPPED_PATH, "a", encoding="utf-8") as f:
        f.write(word + "\n")
      _remove_word_from_pending_file(word)
      return

    # Validate input.
    seen = set()
    valid = True
    for c in user_input:
      if not c.isdigit() or int(c) > len(flashcards) or int(c) == 0 or int(c) in seen:
        print(f"Invalid input: {c}")
        valid = False
        break
      seen.add(int(c))
    if not valid:
      continue

    # Create flashcards based on user input.
    for c in user_input:
      print(f"Creating flashcard: [{c}] {word}")
      flashcard = flashcards[int(c) - 1]
      if is_kana(word):
        model_id = basic_model_id
        fields = {_BASIC_JAPANESE_FIELD: flashcard["example_sentence"], _BASIC_ENGLISH_FIELD: flashcard["english"]}
      else:
        model_id = cloze_model_id
        fields = {
            _CLOZE_EXAMPLE_SENTENCE_FIELD: flashcard["example_sentence"],
            _CLOZE_ENGLISH_FIELD: flashcard["english"]
        }

      _add_flashcard_to_deck(anki_collection, deck_id, model_id, fields)

    # Append word to completed file.
    with open(_WORDS_DONE_PATH, "a", encoding="utf-8") as f:
      f.write(word + "\n")
    _remove_word_from_pending_file(word)

    return


def _add_flashcard_to_deck(anki_collection: Collection, deck_id, model_id, fields: dict[str, str]):
  note = Note(anki_collection, model_id)
  for field, value in fields.items():
    note[field] = value
  anki_collection.add_note(note, deck_id)


def _main():
  # Load Anki collection.
  anki_collection = Collection(os.path.expanduser(_COLLECTION_PATH))

  deck_id = anki_collection.decks.id_for_name(_DECK_NAME)
  if not deck_id:
    raise ValueError(f"Deck name {_DECK_NAME} not found.")
  cloze_model_id = anki_collection.models.id_for_name(_CLOZE_MODEL_NAME)
  if not cloze_model_id:
    raise ValueError(f"Cloze model name (note type) {_CLOZE_MODEL_NAME} not found.")
  basic_model_id = anki_collection.models.id_for_name(_BASIC_MODEL_NAME)
  if not basic_model_id:
    raise ValueError(f"Basic model name (note type) {_BASIC_MODEL_NAME} not found.")
  print(
      f"Loaded Anki collection. Deck ID: {deck_id}, Basic Model ID: {basic_model_id}, Cloze Model ID: {cloze_model_id}")

  constant_words = _get_new_words_from_multiline_string(_NEW_WORDS)
  pending_words = _load_pending_words_from_file()
  if constant_words:
    _append_new_pending_words_to_file(constant_words, pending_words)
    # Refresh pending words.
    pending_words = _load_pending_words_from_file()

  # Make sure there are pending words.
  if not pending_words:
    print("No words to process.")
    return

  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Create future for requesting first word flashcards.
    completion_future = executor.submit(_request_flashcards_from_llm, pending_words[0])

    # Begin interactively generating cards for each word.
    for i in range(len(pending_words)):
      completion = completion_future.result()

      # Request flashcards for the next word, if available.
      if i + 1 < len(pending_words):
        completion_future = executor.submit(_request_flashcards_from_llm, pending_words[i + 1])

      _create_flashcard_interactively(pending_words[i], completion, anki_collection, deck_id, cloze_model_id,
                                      basic_model_id)


if __name__ == "__main__":
  _main()
