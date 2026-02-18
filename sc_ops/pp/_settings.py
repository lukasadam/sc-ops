from dynaconf import Dynaconf

def import_settings(settings_files):
    """Import settings from the specified files.

    Parameters
    ----------
    settings_files
        List of file paths to load settings from.

    Returns
    -------
    Dynaconf
        A Dynaconf instance with the loaded settings.
    """
    return Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=settings_files,
    )