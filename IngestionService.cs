using System.Security.Cryptography;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

public class IngestionException(string message) : Exception(message);

public record IngestResult(string FileName, string Status, int Pages, int Chunks, int TotalChunksStored);

/// <summary>
/// PDF ingestion: hash → extract per page → chunk → embed → store.
/// </summary>
public class IngestionService(
    IEmbeddingClient embedder, IVectorStore store, IConfiguration config, ILogger<IngestionService> logger)
{
    public async Task<IngestResult> IngestPdfAsync(Stream pdfStream, string fileName, CancellationToken ct = default)
    {
        using var ms = new MemoryStream();
        await pdfStream.CopyToAsync(ms, ct);
        var bytes = ms.ToArray();

        // 1. Skip exact duplicates: same bytes = same SHA-256 hash.
        //    Duplicates would crowd other relevant chunks out of the top-k results and waste tokens.
        var hash = Convert.ToHexString(SHA256.HashData(bytes));
        if (store.HasDocument(hash))
            return new IngestResult(fileName, "Skipped: this exact document is already ingested", 0, 0, store.Count);

        // 2. Extract and chunk page by page so every chunk keeps its page number (for citations)
        int chunkSize = config.GetValue("Rag:ChunkSizeWords", 200);
        int overlap = config.GetValue("Rag:ChunkOverlapWords", 40);

        using var pdf = PdfDocument.Open(bytes);
        int pages = pdf.NumberOfPages;

        var pieces = pdf.GetPages()
            .SelectMany(page => TextChunker.Chunk(ContentOrderTextExtractor.GetText(page), chunkSize, overlap)
                .Select(text => (Page: page.Number, Text: text)))
            .ToList();

        if (pieces.Count == 0)
            throw new IngestionException("No text found in the PDF. Scanned PDFs need OCR first.");

        // 3. Embed BEFORE touching the store, so a failed API call never deletes the previous version
        var embeddings = await embedder.EmbedDocumentsAsync(pieces.Select(p => p.Text).ToList(), ct);
        if (embeddings.Count != pieces.Count)
            throw new IngestionException("Embedding count did not match chunk count.");

        var chunks = pieces
            .Zip(embeddings, (p, e) => new DocumentChunk(Guid.NewGuid(), fileName, hash, p.Page, p.Text, e))
            .ToList();

        // 4. Same file name with new content = an updated version: replace old chunks atomically
        int removed = store.ReplaceDocument(fileName, chunks);

        logger.LogInformation(
            "Ingested {FileName}: Pages={Pages} Chunks={Chunks} ReplacedChunks={Removed}",
            fileName, pages, chunks.Count, removed);

        var status = removed > 0 ? "Updated: replaced the previous version" : "Ingested";
        return new IngestResult(fileName, status, pages, chunks.Count, store.Count);
    }
}
