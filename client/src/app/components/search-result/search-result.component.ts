import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { ISearchResults } from '../../entities/search-results.interface.js';
import { SearchService } from '../../services/search.service.js';
import { PagerService } from '../../services/pager.service.js';
import { ArticleService } from '../../services/article.service.js';

@Component({
    selector: 'ss-search-result',
    templateUrl: 'src/app/components/search-result/search-result.component.html',
    styleUrls: ['src/app/components/search-result/search-result.component.css']
})

export class SearchResultComponent implements OnInit {
    search_results: ISearchResults = null
    search_token: string = ''
    pager: any = {};
    resultPerPage: number = 10;

    constructor(private _pagerService: PagerService,
        private _searchService: SearchService, private _articleService: ArticleService,
        private _route: ActivatedRoute) {
        _route.params.subscribe(params => {
            this.search_token = this._route.snapshot.params['term'];
            this._searchService.search(this.search_token)
                .map(response => <ISearchResults>response)
                .subscribe(
                results => { this.showResults(results); },
                error => console.log(error)
                );
        });
    }

    ngOnInit() {
        this.search_token = this._route.snapshot.params['term'];
        this._searchService.search(this.search_token)
            .map (response => <ISearchResults>response)
            .subscribe (
                results => { this.showResults(results);},
                error => console.log(error)
            );
    }

    private showResults(search_results: ISearchResults) {
        this.search_results = search_results;
        this.setPage(1);
    }

    private pageClicked(page: number) {
        this.setPage(page);
        let maxResults = 0;

        if (this.resultPerPage * page > this.search_results.numResults) {
            maxResults = this.search_results.numResults % this.resultPerPage
        }
        else {
            maxResults = 10
        }
        this.search(page, maxResults);
    }

    private setPage(page: number) {
        if (page < 1 || page > this.pager.totalPages) {
            return;
        }
        this.pager = this._pagerService.getPager(this.search_results.numResults, page);
    }

    private search(offset: number, maxResults: number) {
        this._searchService.search(this.search_token, offset, maxResults)
            .map(response => <ISearchResults>response)
            .subscribe(
            results => this.showResults(results),
            error => console.log(error)
            );
    }

    private viewPdf(id: string, download: boolean, title?: string) {
        var saveAs = require('file-saver');
        this._articleService.getArticle(id, true)
            .subscribe(
            results => {
                if (download) {
                    saveAs(results, title + ".pdf");
                    return;
                }
                var fileURL = URL.createObjectURL(results);
                window.open(fileURL);
            },
            error => console.log(error)
            );
    }
}