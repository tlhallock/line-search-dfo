import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.LinkedList;

public class FindOutputLines {

	public static void main(String[] args) throws IOException {
		java.nio.file.Files
				.walkFileTree(
						Paths.get("/home/thallock/Documents/Source/matlab.git/line-search-dfo/"),
						new FileVisitor<Path>() {
							@Override
							public FileVisitResult preVisitDirectory(Path dir,
									BasicFileAttributes attrs)
									throws IOException {
								return FileVisitResult.CONTINUE;
							}

							@Override
							public FileVisitResult visitFile(Path file,
									BasicFileAttributes attrs)
									throws IOException {
								if (!Files.isReadable(file))
									return FileVisitResult.CONTINUE;
								if (!file.toFile().toString().endsWith(".m"))
									return FileVisitResult.CONTINUE;
								System.out.println(file + ":");
								MatlabFile lines = readFile(file);
								lines = preprocess(lines);
								findOutputLines(lines);
								
								
								return FileVisitResult.CONTINUE;
							}

							@Override
							public FileVisitResult visitFileFailed(Path file,
									IOException exc) throws IOException {
								return FileVisitResult.CONTINUE;
							}

							@Override
							public FileVisitResult postVisitDirectory(Path dir,
									IOException exc) throws IOException {
								return FileVisitResult.CONTINUE;
							}
						});
	}

	private static MatlabFile readFile(Path file) throws IOException {
		MatlabFile list = new MatlabFile(file.toFile().toString());
		try (BufferedReader newBufferedReader = Files.newBufferedReader(file);) {
			int idx = 0;
			String line;
			while ((line = newBufferedReader.readLine()) != null) {
				list.add(new Line(line.trim(), idx++));
			}
		}

		return list;
	}

	private static MatlabFile preprocess(MatlabFile matlabCode) {
		MatlabFile returnVal = new MatlabFile(matlabCode.path);

		StringBuilder currentLine = new StringBuilder();
		int currentLineNumber = 0;

		for (Line line : matlabCode) {
			// check for comments

			String uncommented = null;
			int commentIndex = line.theLine.indexOf('%');
			if (commentIndex >= 0) {
				uncommented = line.theLine.substring(0, commentIndex);
			} else {
				uncommented = line.theLine;
			}

			// check for carried lines
			String trimmed = uncommented.trim();

			int indexOf = trimmed.indexOf("...");
			if (indexOf >= 0) {
				currentLine.append(trimmed);
				continue;
			}

			if (trimmed.length() > 0) {
				returnVal.add(new Line(currentLine.append(trimmed)
						.toString(), currentLineNumber));
			}
			currentLine.setLength(0);
			currentLineNumber = line.lineNumber + 1;
		}

		return returnVal;
	}

	private static void findOutputLines(MatlabFile file) {
		for (Line line : file) {
			if (line.hasOutput())
				System.out.println(line);
		}
	}

	private static void printFile(MatlabFile lines) {
		for (Line line : lines) {
			System.out.println(line);
		}
	}

	private static final class MatlabFile extends LinkedList<Line> {
		String path;

		MatlabFile(String path) {
			this.path = path;
		}
	}

	private static final class Line {
		String theLine;
		int lineNumber;

		public Line(String theLine, int lineNumber) {
			this.theLine = theLine;
			this.lineNumber = lineNumber;
		}

		boolean hasOutput() {
			if (theLine.endsWith(";")) {
				return false;
			}

			for (String str : CONTROL_STATEMENTS) {
				if (theLine.contains(str))
					return false;
			}
			
			if (theLine.trim().length() == 0)
				return false;

			return true;
		}

		public String toString() {
			StringBuilder builder = new StringBuilder();

			builder.append(lineNumber).append(":\t\"");
			builder.append(theLine).append("\"\n");

			return builder.toString();
		}
	}

	private static String[] CONTROL_STATEMENTS = new String[] { 
		"end",
		"if",
		"else",
		"while",
		"for", 
		"function",
		"switch",
		"case",
		"otherwise",
		"global"
		};
}
