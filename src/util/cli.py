import sys


def yes_no_prompt(question):
    """ Yes/No confirmation prompt via input(), returns True when answer is yes, False when not.
        If use simply enters, take this as confirmation, we repeatedly ask until we get a valid response.

        Args:
            question: String question to ask the user.
        Returns:
            True for "yes" or False for "no".
    """
    valid = {'yes', 'y', 'ye', 'yup'}
    invalid = {'no', 'n', 'nope'}
    prompt = ' [Y/n] '

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == '' or choice in valid:
            return True
        elif choice in invalid:
            return False
