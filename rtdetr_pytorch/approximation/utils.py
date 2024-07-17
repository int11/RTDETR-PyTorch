import contextlib


print_shape = False
save_variable = False

@contextlib.contextmanager
def saveing():
    global save_variable
    old_value = save_variable
    save_variable = True
    try:
        yield
    finally:
        save_variable = old_value