"""A simple and lean logging framework."""

import sys
from datetime import datetime
from abc import ABC, abstractmethod

import termcolor

class Logger(ABC):
    """Represents the abstract base class for all loggers."""

    @abstractmethod
    def log_info(self, content: str) -> None:
        """
        Outputs informational content to the log.

        Parameters:
        -----------
            content:
                The content that is to be logged.
        """

        raise NotImplementedError()

    @abstractmethod
    def log_success(self, content: str) -> None:
        """
        Outputs success information to the log.

        Parameters:
        -----------
            content:
                The content that is to be logged.
        """

        raise NotImplementedError()

    @abstractmethod
    def log_error(self, content: str) -> None:
        """
        Outputs error information to the log.

        Parameters:
        -----------
            content:
                The content that is to be logged.
        """

        raise NotImplementedError()

    @abstractmethod
    def set_verbosity(self, verbosity: str) -> None:
        """
        Sets the verbosity level.

        Parameters:
        -----------
            verbosity:
                The new verbosity level. Must be either "none", error", "success", or "info".
        """

        raise NotImplementedError()

    @abstractmethod
    def get_verbosity(self) -> str:
        """
        Gets the current verbosity level.

        Returns:
        --------
            Returns the current verbosity level. Can be either "none", error", "success", or "info".
        """

        raise NotImplementedError()

class ConsoleLogger(Logger):
    """Represents a logger, which outputs its logs to the console."""

    def __init__(self) -> None:
        """Initializes a new ConsoleLogger instance."""

        self.verbosity_map = {
            'none': 0,
            'error': 1,
            'success': 2,
            'info': 3
        }
        self.verbosity_level = self.verbosity_map['info']

    def log_info(self, content: str) -> None:
        """
        Outputs informational content to the log.

        Parameters:
        -----------
            content:
                The content that is to be logged.
        """

        if self.verbosity_level >= self.verbosity_map['info']:
            self.log('info', 'cyan', content)

    def log_success(self, content: str) -> None:
        """
        Outputs success information to the log.

        Parameters:
        -----------
            content:
                The content that is to be logged.
        """

        self.log('okay', 'green', content)

    def log_error(self, content: str) -> None:
        """
        Outputs error information to the log.

        Parameters:
        -----------
            content:
                The content that is to be logged.
        """

        if self.verbosity_level >= self.verbosity_map['error']:
            self.log('fail', 'red', content, output_stream=sys.stderr)

    def log(self, log_type: str, log_color: str, content: str, output_stream=sys.stdout) -> None:
        """
        Logs the specified content to the console.

        Parameters:
        -----------
            log_type:
                The type of the log, which is prepended to the actual log.
            log_color:
                The color in which the type of the log is printed out.
            content:
                The content that is to be logged.
            output_stream:
                The stream to which the log is written to. Defaults to standard output, but can, for example, be standard error.
        """

        print('{0} {1}: {2}'.format(
            termcolor.colored('[{0}]'.format(log_type), log_color),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            content.strip()
        ), file=output_stream)

    def set_verbosity(self, verbosity: str) -> None:
        """
        Sets the verbosity level.

        Parameters:
        -----------
            verbosity:
                The new verbosity level. Must be either "none", error", "success", or "info".
        """

        if verbosity not in self.verbosity_map:
            raise ValueError('"{0}" is not a valid verbosity level. Must be either "none", error", "success", or "info".')
        self.verbosity_level = self.verbosity_map[verbosity]

    def get_verbosity(self) -> str:
        """
        Gets the current verbosity level.

        Returns:
        --------
            Returns the current verbosity level. Can be either "none", error", "success", or "info".
        """

        reverse_verbosity_map = {level: verbosity for (verbosity, level) in self.verbosity_map.items()}
        return reverse_verbosity_map[self.verbosity_level]
