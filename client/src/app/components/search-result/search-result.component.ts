import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ISearchResults } from '../../entities/search-results.interface.js';
import { SharedService } from '../../services/shared.service.js';
import { SearchService } from '../../services/search.service.js';
import { PagerService } from '../../services/pager.service.js';

@Component({
    selector: 'ss-search-result',
    templateUrl: 'src/app/components/search-result/search-result.component.html',
    styleUrls: ['src/app/components/search-result/search-result.component.css']
})

export class SearchResultComponent {
    search_results: ISearchResults = null
    search_token: string = ''
    pager: any = {};
    resultPerPage: number = 10;

    constructor(private _sharedService: SharedService, private _pagerService: PagerService, private _searchService: SearchService){ 
        this._sharedService.searchResult$.subscribe (
            results => this.showResults(results),
            error => console.log(error)
        )
        this._sharedService.searchterm$.subscribe (
            result => this.search_token = result,
            error => console.log(error)
        )        
    }

    private showResults(search_results: ISearchResults) {        
        this.search_results = search_results;
        this.setPage(1);
    }

    private pageClicked(page: number) {
        this.setPage(page);
        let maxResults = 0;

        if(this.resultPerPage * page > this.search_results.numResults) {
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
}