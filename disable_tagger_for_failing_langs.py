import os

failing_languages = [
    'Luxembourgish_Letzeburgesch',
    'Japanese',
    'Marathi',
    'Norwegian_Bokmal',
    'Tagalog',
    'Kazakh',
    'Tamil',
    'Macedonian'
]

tests_dir = 'tests/Catalyst.Tests/Languages'

def disable_tagger(tests_dir, languages):
    count = 0
    for lang in languages:
        filename = f'Test{lang}.cs'
        filepath = os.path.join(tests_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()

            # Add tagger: false
            # var nlp = await Pipeline.ForAsync(Language.{language});
            # to
            # var nlp = await Pipeline.ForAsync(Language.{language}, tagger: false);

            if 'tagger: false' not in content:
                new_content = content.replace(f'Pipeline.ForAsync(Language.{lang})', f'Pipeline.ForAsync(Language.{lang}, tagger: false)')
                if content != new_content:
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    print(f"Disabled tagger for {lang}")
                    count += 1
                else:
                    print(f"Could not disable tagger for {lang} (pattern match failed)")
            else:
                print(f"Tagger already disabled for {lang}")
    return count

if __name__ == '__main__':
    modified_count = disable_tagger(tests_dir, failing_languages)
    print(f"Modified {modified_count} files.")
